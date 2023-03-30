"""
Data structure for particle filters
- `effectiveSampleSize(pc :: ParticleContainer)`: Return the effective sample size of the particles in `pc`
"""
mutable struct ParticleContainer{T<:Particle,R<:TracedRNG}
    "Particles."
    vals::Vector{T}
    "Unnormalized logarithmic weights."
    logWs::Vector{Float64}
    "Traced RNG to replay the resampling step"
    rng::R
end

function ParticleContainer(particles::Vector{<:Particle})
    return ParticleContainer(particles, zeros(length(particles)), TracedRNG())
end

function ParticleContainer(particles::Vector{<:Particle}, rng::TracedRNG)
    return ParticleContainer(particles, zeros(length(particles)), rng)
end

function ParticleContainer(
    particles::Vector{<:Particle}, trng::TracedRNG, rng::Random.AbstractRNG
)
    pc = ParticleContainer(particles, trng)
    return seed_from_rng!(pc, rng)
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

    # Copy rng and states
    rng = copy(pc.rng)

    return ParticleContainer(vals, logWs, rng)
end

"""
    update_ref!(particle::Trace, pc::ParticleContainer)

Update reference trajectory. Defaults to `nothing`
"""
update_ref!(particle::Trace, pc::ParticleContainer) = nothing

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
    update_keys!(pc::ParticleContainer)

Create new unique keys for the particles in the ParticleContainer
"""
function update_keys!(pc::ParticleContainer, ref::Union{Particle,Nothing}=nothing)
    # Update keys to new particle ids
    nparticles = length(pc)
    n = ref === nothing ? nparticles : nparticles - 1
    for i in 1:n
        pi = pc.vals[i]
        k = split(state(pi.rng.rng))
        Random.seed!(pi.rng, k[1])
    end
    return nothing
end

"""
    seed_from_rng!(pc::ParticleContainer, rng::Random.AbstractRNG, ref::Union{Particle,Nothing}=nothing)

Set seeds of particle rng from user-provided `rng`
"""
function seed_from_rng!(
    pc::ParticleContainer{T,<:TracedRNG{R,N,<:Random123.AbstractR123{I}}},
    rng::Random.AbstractRNG,
    ref::Union{Particle,Nothing}=nothing,
) where {T,R,N,I}
    n = length(pc.vals)
    nseeds = isnothing(ref) ? n : n - 1

    sampler = Random.Sampler(rng, I)
    for i in 1:nseeds
        subrng = pc.vals[i].rng
        Random.seed!(subrng, gen_seed(rng, subrng, sampler))
    end
    Random.seed!(pc.rng, gen_seed(rng, pc.rng, sampler))

    return pc
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
    ::Random.AbstractRNG,
    pc::ParticleContainer,
    randcat=DEFAULT_RESAMPLER,
    ref::Union{Particle,Nothing}=nothing;
    weights=getweights(pc),
)
    # sample ancestor indices
    n = length(pc)
    nresamples = ref === nothing ? n : n - 1
    indx = randcat(pc.rng, weights, nresamples)

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

            key = isref ? safe_get_refseed(ref.rng) : state(p.rng.rng) # Pick up the alternative rng stream if using the reference particle
            nsplits = isref ? ni + 1 : ni # We need one more seed to refresh the alternative rng stream
            seeds = split(key, nsplits)
            isref && safe_set_refseed!(ref.rng, seeds[end]) # Refresh the alternative rng stream

            Random.seed!(p.rng, seeds[1])

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
        update_ref!(ref, pc)
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
    else
        update_keys!(pc, ref)
    end

    return pc
end

"""
    reweight!(pc::ParticleContainer)

Check if the final time step is reached, and otherwise reweight the particles by
considering the next observation.
"""
function reweight!(pc::ParticleContainer, ref::Union{Particle,Nothing}=nothing)
    n = length(pc)

    particles = collect(pc)
    numdone = 0
    for i in 1:n
        p = particles[i]

        # Obtain ``\\log p(yₜ | y₁, …, yₜ₋₁, x₁, …, xₜ, θ₁, …, θₜ)``, or `nothing` if the
        # the execution of the model is finished.
        # Here ``yᵢ`` are observations, ``xᵢ`` variables of the particle filter, and
        # ``θᵢ`` are variables of other samplers.
        isref = p === ref
        score = advance!(p, isref)

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
    isdone = reweight!(pc, ref)

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
        isdone = reweight!(pc, ref)

        # Compute the normalizing constant ``Z₁`` after reweighting.
        logZ1 = logZ(pc)

        # Compute the estimate of the log evidence ``\\log p(y₁, …, yₜ)``.
        logevidence += logZ1 - logZ0
    end

    return logevidence
end
