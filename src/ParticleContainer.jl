
"""
Data structure for particle filters
- effectiveSampleSize(pc :: ParticleContainer)
- normalise!(pc::ParticleContainer)
- consume(pc::ParticleContainer): return incremental likelihood
"""
mutable struct ParticleContainer{T} <: AbstractParticleContainer where T<:Trace
    vals::Vector{T}
    # A named tuple with functions to manipulate the particles vi.
    logWs::Vector{Float64}
    #This corresponds to log p(x_{0:t}), therefore the log likelihood of the transitions up to this point.
    #Especially important for ancestor sampling.
    logpseq::Vector{Float64}
    # log model evidence
    logE::Float64
    # helpful for rejuvenation steps, e.g. in SMC2
    n_consume::Int
end


Base.collect(pc :: ParticleContainer) = pc.vals # prev: Dict, now: Array
Base.length(pc :: ParticleContainer)  = length(pc.vals)
# pc[i] returns the i'th particle
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]



# registers a new x-particle in the container
function Base.push!(pc::ParticleContainer, p::Particle)
    push!(pc.vals, p)
    push!(pc.logWs, 0.0)
    push!(pc.logpseq,0.0)
    return pc
end

# clears the container but keep params, logweight etc.
function Base.empty!(pc::ParticleContainer)
    pc.vals  = eltype(pc.vals)[]
    pc.logWs = Float64[]
    pc.logpseq = Float64[]
    return pc
end

# clones a theta-particle
function Base.copy(pc::ParticleContainer, utility_functions<:AbstractPFUtilitFunctions)
    # fork particles
    vals = eltype(pc.vals)[fork(p, utility_functions.copy) for p in pc.vals]
    # copy weights
    logWs = copy(pc.logWs)
    logpseq = copy(pc.logpseq)
    ParticleContainer( vals, logWs, logpseq, pc.logE, pc.n_consume)
end

# run particle filter for one step, return incremental likelihood
function Libtask.consume(pc::ParticleContainer)
    # normalisation factor: 1/N
    z1 = logZ(pc)
    n = length(pc)

    particles = collect(pc)
    num_done = 0
    for i=1:n
        p = particles[i]
        score = Libtask.consume(p)
        if hasproperty(typeof(p.taskinfo), logpseq)
            set_logpseq!(pc, i, p.taskinfo.logpseq)
        end
        if score isa Real
            score += p.taskinfo.logp
            reset_logp!(p.taskinfo)
            increase_logweight(pc, i, Float64(score))
        elseif score == Val{:done}
            num_done += 1
        else
            println(score)
            error("[consume]: error in running particle filter.")
        end
    end

    if num_done == n
        res = Val{:done}
    elseif num_done != 0
        error("[consume]: mis-aligned execution traces, num_particles= $(n), num_done=$(num_done).")
    else
        # update incremental likelihoods
        z2 = logZ(pc)
        res = increase_logevidence(pc, z2 - z1)
        pc.n_consume += 1
        # res = increase_loglikelihood(pc, z2 - z1)
    end
    return res
end


# compute the normalized weights
weights(pc::ParticleContainer) = softmax(pc.logWs)

# compute the log-likelihood estimate, ignoring constant term ``- \log num_particles``
logZ(pc::ParticleContainer) = logsumexp(pc.logWs)

# compute the effective sample size ``1 / ∑ wᵢ²``, where ``wᵢ```are the normalized weights
function effectiveSampleSize(pc :: ParticleContainer)
    Ws = weights(pc)
    return inv(sum(abs2, Ws))
end

increase_logweight!(pc::ParticleContainer, t::Int, logw::Float64) = (pc.logWs[t]  += logw)

set_logpseq!(pc::ParticleContainer, t::Int, logp::Float64) = (pc.logpseq[t] = logp)

increase_logevidence!(pc::ParticleContainer, logw::Float64) = (pc.logE += logw)
