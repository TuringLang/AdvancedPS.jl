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
function transition_logweight(particle::SSMTrace, x; kwargs...)
    iter = current_step(particle) - 1
    score = SSMProblems.logdensity(
        dynamics(particle.model, iter), iter, particle.model.X[iter - 1], x, kwargs...
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
            new_state = SSMProblems.simulate(particle.rng, dynamics(model, running_step))
        else
            current_state = model.X[running_step - 1]
            new_state = SSMProblems.simulate(
                particle.rng, dynamics(model, running_step), running_step, current_state
            )
        end
    else
        # We need the current state from the reference particle
        new_state = model.X[running_step]
    end

    score = SSMProblems.logdensity(
        observation(model, running_step), running_step, new_state, model.Y[running_step]
    )

    # Accept transition and move the time index/rng counter
    !isref && push!(model.X, new_state)
    inc_counter!(particle.rng)

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

function update_ref!(ref::SSMTrace, pc::ParticleContainer{<:SSMTrace}, sampler::PGAS)
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
