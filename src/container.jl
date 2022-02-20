struct Trace{F}
    f::F
    task::Libtask.TapedTask
end

const Particle = Trace

function Trace(f)
    if hasfield(typeof(f), :evaluator) # Test whether f is a Turing.TracedModel
        # println(f.evaluator)
        task = Libtask.TapedTask(f.evaluator[1], f.evaluator[2:end]...)
    else # f is a Function, or AdavncedPS.Model
        task = Libtask.TapedTask(f)
    end

    # add backward reference
    newtrace = Trace(f, task)
    addreference!(task.task, newtrace)

    return newtrace
end

Base.copy(trace::Trace) = Trace(trace.f, copy(trace.task))

# step to the next observe statement and
# return the log probability of the transition (or nothing if done)
advance!(t::Trace) = Libtask.consume(t.task)

# reset log probability
reset_logprob!(t::Trace) = nothing

reset_model(f) = deepcopy(f)
delete_retained!(f) = nothing

# Task copying version of fork for Trace.
function fork(trace::Trace, isref::Bool=false)
    newtrace = copy(trace)
    isref && delete_retained!(newtrace.f)

    # add backward reference
    addreference!(newtrace.task.task, newtrace)

    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace::Trace)
    newf = reset_model(trace.f)
    # task = Libtask.TapedTask(trace.task)    
    if hasfield(typeof(newf), :evaluator) # Test whether f is a Turing.TracedModel
        task = Libtask.TapedTask(newf.evaluator[1], newf.evaluator[2:end]...)
    else # f is a Function, or AdavncedPS.Model
        task = Libtask.TapedTask(newf)
    end

    # add backward reference
    newtrace = Trace(newf, task)
    addreference!(task.task, newtrace)

    return newtrace
end

# create a backward reference in task_local_storage
function addreference!(task::Task, trace::Trace)
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:__trace] = trace

    return task
end

current_trace() = current_task().storage[:__trace]

"""
Data structure for particle filters
- effectiveSampleSize(pc :: ParticleContainer)
- normalise!(pc::ParticleContainer)
- consume(pc::ParticleContainer): return incremental likelihood
"""
mutable struct ParticleContainer{T<:Particle}
    "Particles."
    vals::Vector{T}
    "Unnormalized logarithmic weights."
    logWs::Vector{Float64}
end

function ParticleContainer(particles::Vector{<:Particle})
    return ParticleContainer(particles, zeros(length(particles)))
end

Base.collect(pc::ParticleContainer) = pc.vals
Base.length(pc::ParticleContainer) = length(pc.vals)
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]

function Base.rand(rng::Random.AbstractRNG, pc::ParticleContainer)
    index = randcat(rng, getweights(pc))
    return pc[index]
end

# registers a new x-particle in the container
function Base.push!(pc::ParticleContainer, p::Particle)
    push!(pc.vals, p)
    push!(pc.logWs, 0.0)
    return pc
end

# clones a theta-particle
function Base.copy(pc::ParticleContainer)
    # fork particles
    vals = eltype(pc.vals)[fork(p) for p in pc.vals]

    # copy weights
    logWs = copy(pc.logWs)

    return ParticleContainer(vals, logWs)
end

"""
    reset_logweights!(pc::ParticleContainer)

Reset all unnormalized logarithmic weights to zero.
"""
function reset_logweights!(pc::ParticleContainer)
    fill!(pc.logWs, 0.0)
    return pc
end

"""
    increase_logweight!(pc::ParticleContainer, i::Int, x)

Increase the unnormalized logarithmic weight of the `i`th particle with `x`.
"""
function increase_logweight!(pc::ParticleContainer, i, logw)
    pc.logWs[i] += logw
    return pc
end

"""
    getweights(pc::ParticleContainer)

Compute the normalized weights of the particles.
"""
getweights(pc::ParticleContainer) = StatsFuns.softmax(pc.logWs)

"""
    getweight(pc::ParticleContainer, i)

Compute the normalized weight of the `i`th particle.
"""
getweight(pc::ParticleContainer, i) = exp(pc.logWs[i] - logZ(pc))

"""
    logZ(pc::ParticleContainer)

Return the logarithm of the normalizing constant of the unnormalized logarithmic weights.
"""
logZ(pc::ParticleContainer) = StatsFuns.logsumexp(pc.logWs)

"""
    effectiveSampleSize(pc::ParticleContainer)

Compute the effective sample size ``1 / ∑ wᵢ²``, where ``wᵢ```are the normalized weights.
"""
function effectiveSampleSize(pc::ParticleContainer)
    Ws = getweights(pc)
    return inv(sum(abs2, Ws))
end

"""
    resample_propagate!(rng, pc::ParticleContainer[, randcat = resample_systematic,
                        ref = nothing; weights = getweights(pc)])

Resample and propagate the particles in `pc`.

Function `randcat` is used for sampling ancestor indices from the categorical distribution
of the particle `weights`. For Particle Gibbs sampling, one can provide a reference particle
`ref` that is ensured to survive the resampling step.
"""
function resample_propagate!(
    rng::Random.AbstractRNG,
    pc::ParticleContainer,
    randcat=resample_systematic,
    ref::Union{Particle,Nothing}=nothing;
    weights=getweights(pc),
)
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
            children[j += 1] = p

            # fork additional children
            for _ in 2:ni
                children[j += 1] = fork(p, isref)
            end
        end
    end

    if ref !== nothing
        # Insert the retained particle. This is based on the replaying trick for efficiency
        # reasons. If we implement PG using task copying, we need to store Nx * T particles!
        @inbounds children[n] = ref
    end

    # replace particles and log weights in the container with new particles and weights
    pc.vals = children
    reset_logweights!(pc)

    return pc
end

function resample_propagate!(
    rng::Random.AbstractRNG,
    pc::ParticleContainer,
    resampler::ResampleWithESSThreshold,
    ref::Union{Particle,Nothing}=nothing;
    weights=getweights(pc),
)
    # Compute the effective sample size ``1 / ∑ wᵢ²`` with normalized weights ``wᵢ``
    ess = inv(sum(abs2, weights))

    if ess ≤ resampler.threshold * length(pc)
        resample_propagate!(rng, pc, resampler.resampler, ref; weights=weights)
    end

    return pc
end

"""
    reweight!(pc::ParticleContainer)

Check if the final time step is reached, and otherwise reweight the particles by
considering the next observation.
"""
function reweight!(pc::ParticleContainer)
    n = length(pc)

    particles = collect(pc)
    numdone = 0
    for i in 1:n
        p = particles[i]

        # Obtain ``\\log p(yₜ | y₁, …, yₜ₋₁, x₁, …, xₜ, θ₁, …, θₜ)``, or `nothing` if the
        # the execution of the model is finished.
        # Here ``yᵢ`` are observations, ``xᵢ`` variables of the particle filter, and
        # ``θᵢ`` are variables of other samplers.
        score = advance!(p)

        if score === nothing
            numdone += 1
        else
            # Increase the unnormalized logarithmic weights.
            increase_logweight!(pc, i, score)

            # Reset the accumulator of the log probability in the model so that we can
            # accumulate log probabilities of variables of other samplers until the next
            # observation.
            reset_logprob!(p)
        end
    end

    # Check if all particles are propagated to the final time point.
    numdone == n && return true

    # The posterior for models with random number of observations is not well-defined.
    if numdone != 0
        error(
            "mis-aligned execution traces: # particles = ",
            n,
            " # completed trajectories = ",
            numdone,
            ". Please make sure the number of observations is NOT random.",
        )
    end

    return false
end

"""
    sweep!(rng, pc::ParticleContainer, resampler)

Perform a particle sweep and return an unbiased estimate of the log evidence.

The resampling steps use the given `resampler`.

# Reference

Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential monte carlo samplers.
Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(3), 411-436.
"""
function sweep!(
    rng::Random.AbstractRNG,
    pc::ParticleContainer,
    resampler,
    ref::Union{Particle,Nothing}=nothing,
)
    # Initial step:

    # Resample and propagate particles.
    resample_propagate!(rng, pc, resampler, ref)

    # Compute the current normalizing constant ``Z₀`` of the unnormalized logarithmic
    # weights.
    # Usually it is equal to the number of particles in the beginning but this
    # implementation covers also the unlikely case of a particle container that is
    # initialized with non-zero logarithmic weights.
    logZ0 = logZ(pc)

    # Reweight the particles by including the first observation ``y₁``.
    isdone = reweight!(pc)

    # Compute the normalizing constant ``Z₁`` after reweighting.
    logZ1 = logZ(pc)

    # Compute the estimate of the log evidence ``\\log p(y₁)``.
    logevidence = logZ1 - logZ0

    # For observations ``y₂, …, yₜ``:
    while !isdone
        # Resample and propagate particles.
        resample_propagate!(rng, pc, resampler, ref)

        # Compute the current normalizing constant ``Z₀`` of the unnormalized logarithmic
        # weights.
        logZ0 = logZ(pc)

        # Reweight the particles by including the next observation ``yₜ``.
        isdone = reweight!(pc)

        # Compute the normalizing constant ``Z₁`` after reweighting.
        logZ1 = logZ(pc)

        # Compute the estimate of the log evidence ``\\log p(y₁, …, yₜ)``.
        logevidence += logZ1 - logZ0
    end

    return logevidence
end
