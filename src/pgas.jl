""" 
    initialization(model::AbstractStateSpaceModel)

Define the distribution of the initial state of the State Space Model
"""
function initialization end

""" 
    transition(model::AbstractStateSpaceModel)

Define the transition density of the State Space Model
"""
function transition end

"""
    observation(model::AbstractStateSpaceModel)

Return the log-likelihood of the observed measurement conditional on the current state of the model. 
Must return `nothing` if it consumed all the data
"""
function observation end

"""
    inc_step!(model::AbstractStateSpaceModel, n::Int=1)

Increase the internal model counter by `n`. Used to keep track of the current timestep when replaying
"""
inc_step!(m::AbstractStateSpaceModel, n::Int=1) = m.T += n

"""
    reset!(m::AbstractStateSpaceModel)

Set model counter to `1` to restart model evaluation
"""
reset!(m::AbstractStateSpaceModel) = m.T = 1

"""
    previous_state(model::AbstractStateSpaceModel)

Return `Xₜ₋₁` or `nothing` from `model`
"""
function previous_state(model::AbstractStateSpaceModel)
    return model.T < 2 ? nothing : model.X[model.T - 1]
end

"""
    current_step(model::AbstractStateSpaceModel)

Return current model step
"""
current_step(model::AbstractStateSpaceModel) = model.T

"""
    transition_logweight(model::AbstractStateSpaceModel, x)

Get the log weight of the transition from previous state of `model` to `x`
"""
function transition_logweight(particle::SSMTrace, x)
    return Distributions.logpdf(transition(particle.f, previous_state(particle.f)), x)
end

"""
    get_ancestor_logweights(pc::ParticleContainer{F,R}, x) where {F<:SSMTrace,R}

Get the ancestor log weights for each particle in `pc`
"""
function get_ancestor_logweights(pc::ParticleContainer{F,R}, x) where {F<:SSMTrace,R}
    particles = collect(pc.vals)
    nparticles = length(particles)
    weights = pc.logWs

    logweights = map(1:nparticles) do i
        transition_logweight(particles[i], x) + weights[i]
    end

    return logweights
end

""" 
    advance!(particle::SSMTrace, isref::Bool=false)

Return the log-probability of the transition nothing if done
"""
function advance!(particle::SSMTrace, isref::Bool=false)
    isref ? load_state!(particle.rng) : save_state!(particle.rng)

    model = particle.f
    if !isref
        if model.T == 1
            new_state = rand(particle.rng, initialization(model)) # Generate initial state, maybe fallback to 0 is initialization is not defined
        else
            current_state = model.X[model.T - 1]
            new_state = rand(particle.rng, transition(model, current_state))
        end
    else
        new_state = model.X[model.T <= length(model.X) ? model.T : end] # Hacky, reached beyond last time step
    end

    score = observation(model, new_state)

    if score !== nothing # Accept transition
        !isref && push!(model.X, new_state)
        inc_step!(model)
        inc_counter!(particle.rng) # Increase rng counter, don't really need it but keep things consistent
    else
        inc_step!(model, -1)
    end

    return score
end

function truncate!(model::AbstractStateSpaceModel)
    step = model.T <= length(model.X) ? model.T - 1 : length(model.X)
    return model.X = model.X[1:step]
end

function fork(trace::SSMTrace, isref::Bool)
    model = deepcopy(trace.f)
    isref && truncate!(model) # Forget the rest of the reference trajectory
    return SSMTrace(model, trace.rng)
end

function forkr(trace::SSMTrace)
    Random123.set_counter!(trace.rng, 1)
    reset!(trace.f) # Reset model count, we could probably use the TracedRNG inner counter 
    return SSMTrace(deepcopy(trace.f), trace.rng)
end

function update_ancestor!(ref::SSMTrace, pc::ParticleContainer)
    current_step(ref.f) == 1 && return nothing

    particles = collect(pc.vals)
    ancestor_weights = get_ancestor_logweights(pc, ref.f.X[ref.f.T - 1])
    weights = StatsFuns.softmax(ancestor_weights)
    ancestor_index = rand(Distributions.Categorical(weights))
    ancestor = particles[ancestor_index]
    return ref.f.X[ref.f.T - 1] = previous_state(ancestor.f)
end

function resample_propagate!(
    rng::Random.AbstractRNG,
    pc::ParticleContainer{T,R},
    randcat=resample_systematic,
    ref::Union{Particle,Nothing}=nothing;
    weights=getweights(pc),
) where {T<:SSMTrace,R}

    # check that weights are not NaN
    @assert !any(isnan, weights)

    # sample ancestor indices
    n = length(pc)
    nresamples = ref === nothing ? n : n - 1
    indx = randcat(rng, weights, nresamples)

    # count number of children for each particle
    num_children = zeros(Int, n)
    @inbounds for i in indx
        num_children[i] += 1
    end

    # fork particles
    particles = collect(pc)
    children = similar(particles)
    j = 0
    @inbounds for i in 1:n
        ni = num_children[i]

        if ni > 0
            # fork first child
            pi = particles[i]
            isref = pi === ref
            p = isref ? fork(pi, isref) : pi
            nseeds = isref ? ni - 1 : ni

            seeds = split(p.rng.rng.key, nseeds)
            !isref && Random.seed!(p.rng, seeds[1])

            children[j += 1] = p
            # fork additional children
            for k in 2:ni
                part = fork(p, isref)
                Random.seed!(part.rng, seeds[k])
                children[j += 1] = part
            end
        end
    end

    if ref !== nothing
        # Insert the retained particle. This is based on the replaying trick for efficiency
        # reasons. If we implement PG using task copying, we need to store Nx * T particles!
        update_ancestor!(ref, pc)
        @inbounds children[n] = ref
    end

    # replace particles and log weights in the container with new particles and weights
    pc.vals = children
    reset_logweights!(pc)

    return pc
end
