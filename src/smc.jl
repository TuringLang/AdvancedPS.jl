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
    logevidence = sweep!(rng, particles, sampler.resampler)

    return SMCSample(collect(particles), getweights(particles), logevidence)
end

struct PG{R} <: AbstractMCMC.AbstractSampler
    """Number of particles."""
    nparticles::Int
    """Resampling algorithm."""
    resampler::R
end

"""
    PG(n, [resampler = ResampleWithESSThreshold()])
    PG(n, [resampler = resample_systematic, ]threshold)

Create a Particle Gibbs sampler with `n` particles.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
PG(nparticles::Int) = PG(nparticles, ResampleWithESSThreshold())

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real)
    return PG(nparticles, ResampleWithESSThreshold(resampler, threshold))
end
PG(nparticles::Int, threshold::Real) = PG(nparticles, DEFAULT_RESAMPLER, threshold)

struct PGState{T}
    trajectory::T
end

struct PGSample{T,L}
    trajectory::T
    logevidence::L
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractStateSpaceModel,
    sampler::PG,
    state::Union{PGState,Nothing}=nothing;
    kwargs...,
)
    # Create a new set of particles.
    nparticles = sampler.nparticles
    isref = !isnothing(state)

    traces = map(1:nparticles) do i
        if i == nparticles && isref
            # Create reference trajectory.
            forkr(copy(state.trajectory))
        else
            trng = TracedRNG()
            Trace(deepcopy(model), trng)
        end
    end

    particles = ParticleContainer(traces, TracedRNG(), rng)

    # Perform a particle sweep.
    reference = isref ? particles.vals[nparticles] : nothing
    logevidence = sweep!(rng, particles, sampler.resampler, reference)

    # Pick a particle to be retained.
    newtrajectory = rand(rng, particles)

    return PGSample(newtrajectory, logevidence), PGState(newtrajectory)
end

struct PGAS{R} <: AbstractMCMC.AbstractSampler
    """Number of particles."""
    nparticles::Int
    """Resampling algorithm."""
    resampler::R
end

PGAS(nparticles::Int) = PGAS(nparticles, ResampleWithESSThreshold(1.0))

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractStateSpaceModel,
    sampler::PGAS,
    state::Union{PGState,Nothing}=nothing;
    kwargs...,
)
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
    logevidence = sweep!(rng, particles, sampler.resampler, reference)

    # Pick a particle to be retained.
    newtrajectory = rand(particles.rng, particles)
    return PGSample(newtrajectory, logevidence), PGState(newtrajectory)
end
