struct SMC{R} <: AbstractMCMC.AbstractSampler
    nparticles::Int
    resampler::R
end

"""
    SMC(n[, resampler = ResampleWithESSThreshold()])
    SMC(n, [resampler = resample_systematic, ]threshold)

Create a sequential Monte Carlo (SMC) sampler with `n` particles.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
SMC(nparticles::Int) = SMC(nparticles, ResampleWithESSThreshold())

# Convenient constructors with ESS threshold
function SMC(nparticles::Int, resampler, threshold::Real)
    return SMC(nparticles, ResampleWithESSThreshold(resampler, threshold))
end
SMC(nparticles::Int, threshold::Real) = SMC(nparticles, DEFAULT_RESAMPLER, threshold)

struct SMCSample{P,W,L}
    trajectories::P
    weights::W
    logevidence::L
end

function AbstractMCMC.sample(model::AbstractStateSpaceModel, sampler::SMC; kwargs...)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, sampler; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG, model::AbstractStateSpaceModel, sampler::SMC; kwargs...
)
    if !isempty(kwargs)
        @warn "keyword arguments $(keys(kwargs)) are not supported by `SMC`"
    end

    traces = map(1:(sampler.nparticles)) do i
        trng = TracedRNG()
        Trace(deepcopy(model), trng)
    end

    # Create a set of particles.
    particles = ParticleContainer(traces, TracedRNG(), rng)

    # Perform particle sweep.
    logevidence = sweep!(rng, particles, sampler.resampler, IdentityReferenceSampler)

    return SMCSample(collect(particles), getweights(particles), logevidence)
end

struct PG{R,T<:AbstractReferenceSampler} <: AbstractMCMC.AbstractSampler
    """Number of particles."""
    nparticles::Int
    """Resampling algorithm."""
    resampler::R

    PG{T}(nparticles::Int, resampler::R) where {T,R} = new{R,T}(nparticles, resampler)
    PG{R,T}(nparticles::Int, resampler::R) where {T,R} = new{R,T}(nparticles, resampler)
end

"""
    PG(n, [resampler = ResampleWithESSThreshold()])
    PG(n, [resampler = resample_systematic, ]threshold)

Create a Particle Gibbs sampler with `n` particles.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function PG(nparticles::Int)
    let resampler = ResampleWithESSThreshold()
        PG{typeof(resampler),IdentityReferenceSampler}(nparticles, resampler)
    end
end

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real)
    resampler_oject = ResampleWithESSThreshold(resampler, threshold)
    T = typeof(resampler_oject)
    return PG{T,IdentityReferenceSampler}(
        nparticles, ResampleWithESSThreshold(resampler, threshold)
    )
end
PG(nparticles::Int, threshold::Real) = PG(nparticles, DEFAULT_RESAMPLER, threshold)

struct PGState{T}
    trajectory::T
end

struct PGSample{T,L}
    trajectory::T
    logevidence::L
end

const PGAS{R} = PG{R,AncestorReferenceSampler}
function PGAS(nparticles::Int)
    let resampler = ResampleWithESSThreshold(1.0)
        PG{typeof(resampler),AncestorReferenceSampler}(nparticles, resampler)
    end
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractStateSpaceModel,
    sampler::PG{R,T},
    state::Union{PGState,Nothing}=nothing;
    kwargs...,
) where {R,T<:AbstractReferenceSampler}
    # Create a new set of particles.
    nparticles = sampler.nparticles
    isref = !isnothing(state)

    traces = map(1:nparticles) do i
        if i == nparticles && isref
            # Create reference trajectory.
            forkr(deepcopy(state.trajectory))
        else
            Trace(deepcopy(model), TracedRNG())
        end
    end
    particles = ParticleContainer(traces, TracedRNG(), rng)

    # Perform a particle sweep.
    reference = isref ? particles.vals[nparticles] : nothing
    logevidence = sweep!(rng, particles, sampler.resampler, T, reference)

    # Pick a particle to be retained.
    newtrajectory = rand(particles.rng, particles)
    return PGSample(newtrajectory.model, logevidence), PGState(newtrajectory)
end
