
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

# registers a new x-particle in the container
function Base.push!(pc::ParticleContainer, p::Particle)
    push!(pc.vals, p)
    push!(pc.logWs, 0.0)
    pc
end

# clones a theta-particle
function Base.copy(pc::ParticleContainer)
    # fork particles
    vals = eltype(pc.vals)[fork(p) for p in pc.vals]

    # copy weights
    logWs = copy(pc.logWs)

    ParticleContainer(vals, logWs)
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
getweights(pc::ParticleContainer) = softmax(pc.logWs)

"""
    getweight(pc::ParticleContainer, i)
Compute the normalized weight of the `i`th particle.
"""
getweight(pc::ParticleContainer, i) = exp(pc.logWs[i] - logZ(pc))

"""
    logZ(pc::ParticleContainer)
Return the logarithm of the normalizing constant of the unnormalized logarithmic weights.
"""
logZ(pc::ParticleContainer) = logsumexp(pc.logWs)

"""
    effectiveSampleSize(pc::ParticleContainer)
Compute the effective sample size ``1 / ∑ wᵢ²``, where ``wᵢ```are the normalized weights.
"""
function effectiveSampleSize(pc::ParticleContainer)
    Ws = getweights(pc)
    return inv(sum(abs2, Ws))
end
