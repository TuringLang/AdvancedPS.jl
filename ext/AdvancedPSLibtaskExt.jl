module AdvancedPSLibtaskExt

if isdefined(Base, :get_extension)
    using AdvancedPS: AdvancedPS
    using AdvancedPS: Random123
    using AdvancedPS: AbstractMCMC
    using AdvancedPS: Random
    using AdvancedPS: Distributions
    using Libtask: Libtask
else
    using ..AdvancedPS: AdvancedPS
    using ..AdvancedPS: Random123
    using ..AdvancedPS: AbstractMCMC
    using ..AdvancedPS: Random
    using ..AdvancedPS: Distributions
    using ..Libtask: Libtask
end

# In Libtask.TapedTask.taped_globals, this extension sometimes needs to store an RNG,
# and sometimes both an RNG and other information. In Turing.jl the other information
# is a VarInfo. This struct puts those in a single struct. Note the abstract type of
# the second field. This is okay, because `get_taped_globals` needs a type assertion anyway.
struct TapedGlobals{RngType}
    rng::RngType
    other::Any
end

TapedGlobals(rng::Random.AbstractRNG) = TapedGlobals(rng, nothing)

"""
    LibtaskModel{F}

State wrapper to hold `Libtask.CTask` model initiated from `f`.
"""
function AdvancedPS.LibtaskModel(
    f::AdvancedPS.AbstractGenericModel, rng::Random.AbstractRNG, args...
) # Changed the API, need to take care of the RNG properly
    return AdvancedPS.LibtaskModel(f, Libtask.TapedTask(TapedGlobals(rng), f, args...))
end

"""
    copy(model::AdvancedPS.LibtaskModel)

The task is copied (forked) and the inner model is deepcopied.
"""
function Base.copy(model::AdvancedPS.LibtaskModel)
    return AdvancedPS.LibtaskModel(deepcopy(model.f), copy(model.ctask))
end

const LibtaskTrace{R} = AdvancedPS.Trace{<:AdvancedPS.LibtaskModel,R}

function Base.copy(trace::LibtaskTrace)
    newtrace = AdvancedPS.Trace(copy(trace.model), deepcopy(trace.rng))
    set_other_global!(newtrace, newtrace)
    return newtrace
end

"""Get the RNG from a `LibtaskTrace`."""
function get_rng(trace::LibtaskTrace)
    return trace.model.ctask.taped_globals.rng
end

"""Set the RNG for a `LibtaskTrace`."""
function set_rng!(trace::LibtaskTrace, rng::Random.AbstractRNG)
    taped_globals = trace.model.ctask.taped_globals
    Libtask.set_taped_globals!(trace.model.ctask, TapedGlobals(rng, taped_globals.other))
    trace.rng = rng
    return trace
end

"""Set the other "taped global" variable of a `LibtaskTrace`, other than the RNG."""
function set_other_global!(trace::LibtaskTrace, other)
    rng = get_rng(trace)
    Libtask.set_taped_globals!(trace.model.ctask, TapedGlobals(rng, other))
    return trace
end

"""Get the other "taped global" variable of a `LibtaskTrace`, other than the RNG."""
get_other_global(trace::LibtaskTrace) = trace.model.ctask.taped_globals.other

function AdvancedPS.Trace(
    model::AdvancedPS.AbstractGenericModel, rng::Random.AbstractRNG, args...
)
    trace = AdvancedPS.Trace(AdvancedPS.LibtaskModel(model, rng, args...), rng)
    # Set a backreference so that the TapedTask in `trace` stores the `trace` itself in its
    # taped globals.
    set_other_global!(trace, trace)
    return trace
end

# step to the next observe statement and
# return the log probability of the transition (or nothing if done)
function AdvancedPS.advance!(trace::LibtaskTrace, isref::Bool=false)
    rng = get_rng(trace)
    isref ? AdvancedPS.load_state!(rng) : AdvancedPS.save_state!(rng)
    AdvancedPS.inc_counter!(rng)
    # Move to next step
    return Libtask.consume(trace.model.ctask)
end

# Task copying version of fork for Trace.
function AdvancedPS.fork(trace::LibtaskTrace, isref::Bool=false)
    newtrace = copy(trace)
    set_rng!(newtrace, deepcopy(get_rng(newtrace)))
    isref && AdvancedPS.delete_retained!(newtrace.model.f)
    isref && delete_seeds!(newtrace)
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function AdvancedPS.forkr(trace::LibtaskTrace)
    rng = get_rng(trace)
    newf = AdvancedPS.reset_model(trace.model.f)
    Random123.set_counter!(rng, 1)

    ctask = Libtask.TapedTask(TapedGlobals(rng, get_other_global(trace)), newf)
    new_tapedmodel = AdvancedPS.LibtaskModel(newf, ctask)

    # add backward reference
    newtrace = AdvancedPS.Trace(new_tapedmodel, rng)
    AdvancedPS.gen_refseed!(newtrace)
    return newtrace
end

AdvancedPS.update_ref!(::LibtaskTrace) = nothing

"""
    observe(dist::Distribution, x)

Observe sample `x` from distribution `dist` and yield its log-likelihood value.
"""
function AdvancedPS.observe(dist::Distributions.Distribution, x)
    Libtask.produce(Distributions.loglikelihood(dist, x))
    return nothing
end

"""
AbstractMCMC interface. We need libtask to sample from arbitrary callable AbstractModel
"""

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AdvancedPS.AbstractGenericModel,
    sampler::AdvancedPS.PG,
    state::Union{AdvancedPS.PGState,Nothing}=nothing;
    kwargs...,
)
    # Create a new set of particles.
    nparticles = sampler.nparticles
    isref = !isnothing(state)

    traces = map(1:nparticles) do i
        if i == nparticles && isref
            # Create reference trajectory.
            AdvancedPS.forkr(copy(state.trajectory))
        else
            trng = AdvancedPS.TracedRNG()
            trace = AdvancedPS.Trace(deepcopy(model), trng)
            trace
        end
    end

    particles = AdvancedPS.ParticleContainer(traces, AdvancedPS.TracedRNG(), rng)

    # Perform a particle sweep.
    reference = isref ? particles.vals[nparticles] : nothing
    logevidence = AdvancedPS.sweep!(rng, particles, sampler.resampler, sampler, reference)

    # Pick a particle to be retained.
    newtrajectory = rand(rng, particles)

    replayed = AdvancedPS.replay(newtrajectory)
    return AdvancedPS.PGSample(replayed.model.f, logevidence), AdvancedPS.PGState(replayed)
end

function AbstractMCMC.sample(
    model::AdvancedPS.AbstractGenericModel, sampler::AdvancedPS.SMC; kwargs...
)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, sampler; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::AdvancedPS.AbstractGenericModel,
    sampler::AdvancedPS.SMC;
    kwargs...,
)
    if !isempty(kwargs)
        @warn "keyword arguments $(keys(kwargs)) are not supported by `SMC`"
    end

    traces = map(1:(sampler.nparticles)) do i
        trng = AdvancedPS.TracedRNG()
        trace = AdvancedPS.Trace(deepcopy(model), trng)
        trace
    end

    # Create a set of particles.
    particles = AdvancedPS.ParticleContainer(traces, AdvancedPS.TracedRNG(), rng)

    # Perform particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, sampler.resampler, sampler)

    replayed = map(particle -> AdvancedPS.replay(particle).model.f, particles.vals)

    return AdvancedPS.SMCSample(
        collect(replayed), AdvancedPS.getweights(particles), logevidence
    )
end

"""
    replay(particle::AdvancedPS.Particle)

Rewind the particle and regenerate all sampled values. This ensures that the returned trajectory has the correct values.
"""
function AdvancedPS.replay(particle::AdvancedPS.Particle)
    trng = deepcopy(particle.rng)
    Random123.set_counter!(trng.rng, 0)
    trng.count = 1
    trace = AdvancedPS.Trace(
        AdvancedPS.LibtaskModel(deepcopy(particle.model.f), trng), trng
    )
    score = AdvancedPS.advance!(trace, true)
    while !isnothing(score)
        score = AdvancedPS.advance!(trace, true)
    end
    return trace
end

"""
    delete_seeds!(particle::Particle)

Truncate the seed history from the `particle` rng. When forking the reference Particle
we need to keep the seeds up to the current model iteration but generate new seeds
and random values afterward.
"""
function delete_seeds!(particle::AdvancedPS.Particle)
    return particle.rng.keys = particle.rng.keys[1:(particle.rng.count - 1)]
end

end
