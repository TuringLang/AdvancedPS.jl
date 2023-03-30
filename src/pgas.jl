"""
    initialization(model::AbstractStateSpaceModel)

Define the distribution of the initial state of the State Space Model
"""
function initialization end

"""
    transition(model::AbstractStateSpaceModel, state, step)

Define the transition density of the State Space Model
Must return `nothing` if it consumed all the data
"""
function transition end

"""
    observation(model::AbstractStateSpaceModel, state, step)

Return the log-likelihood of the observed measurement conditional on the current state of the model.
Must return `nothing` if it consumed all the data
"""
function observation end

"""
    isdone(model::AbstractStateSpaceModel, step)

Return `true` if model reached final state else `false`
"""
function isdone end

"""
    previous_state(trace::SSMTrace)

Return `Xₜ₋₁` or `nothing` from `model`
"""
function previous_state(trace::SSMTrace)
    return trace.model.X[current_step(trace) - 1]
end

function past_idx(trace::SSMTrace)
    return 1:(current_step(trace) - 1)
end

"""
    current_step(model::AbstractStateSpaceModel)

Return current model step
"""
current_step(trace::SSMTrace) = trace.rng.count

"""
    transition_logweight(model::AbstractStateSpaceModel, x)

Get the log weight of the transition from previous state of `model` to `x`
"""
function transition_logweight(particle::SSMTrace, x)
    score = Distributions.logpdf(
        transition(
            particle.model,
            particle.model.X[current_step(particle) - 2],
            current_step(particle) - 2,
        ),
        x,
    )
    return score
end

"""
    get_ancestor_logweights(pc::ParticleContainer{F,R}, x) where {F<:SSMTrace,R}

Get the ancestor log weights for each particle in `pc`
"""
function get_ancestor_logweights(pc::ParticleContainer{<:SSMTrace}, x, weights)
    nparticles = length(pc.vals)

    logweights = map(1:nparticles) do i
        transition_logweight(pc.vals[i], x) + weights[i]
    end
    return logweights
end

"""
    advance!(particle::SSMTrace, isref::Bool=false)

Return the log-probability of the transition nothing if done
"""
function advance!(particle::SSMTrace, isref::Bool=false)
    isref ? load_state!(particle.rng) : save_state!(particle.rng)

    model = particle.model
    running_step = current_step(particle)
    isdone(model, running_step) && return nothing

    if !isref
        if running_step == 1
            new_state = rand(particle.rng, initialization(model)) # Generate initial state, maybe fallback to 0 if initialization is not defined
        else
            current_state = model.X[running_step - 1]
            new_state = rand(particle.rng, transition(model, current_state, running_step))
        end
    else
        new_state = model.X[running_step] # We need the current state from the reference particle
    end

    score = observation(model, new_state, running_step)

    # accept transition
    !isref && push!(model.X, new_state)
    inc_counter!(particle.rng) # Increase rng counter, we use it as the model `time` index instead of handling two distinct counters

    return score
end

function truncate!(particle::SSMTrace)
    model = particle.model
    idx = past_idx(particle)
    model.X = model.X[idx]
    particle.rng.keys = particle.rng.keys[idx]
    return model
end

function fork(particle::SSMTrace, isref::Bool)
    model = deepcopy(particle.model)
    new_particle = Trace(model, deepcopy(particle.rng))
    isref && truncate!(new_particle) # Forget the rest of the reference trajectory
    return new_particle
end

function forkr(particle::SSMTrace)
    Random123.set_counter!(particle.rng, 1)
    newtrace = Trace(deepcopy(particle.model), deepcopy(particle.rng))
    gen_refseed!(newtrace)
    return newtrace
end

function update_ref!(ref::SSMTrace, pc::ParticleContainer{<:SSMTrace})
    current_step(ref) <= 2 && return nothing # At the beginning of step + 1 since we start at 1
    isdone(ref.model, current_step(ref)) && return nothing

    ancestor_weights = get_ancestor_logweights(
        pc, ref.model.X[current_step(ref) - 1], pc.logWs
    )
    norm_weights = StatsFuns.softmax(ancestor_weights)

    ancestor_index = rand(pc.rng, Distributions.Categorical(norm_weights))
    ancestor = pc.vals[ancestor_index]

    idx = past_idx(ref)
    ref.model.X[idx] = ancestor.model.X[idx]
    return ref.rng.keys[idx] = ancestor.rng.keys[idx]
end
