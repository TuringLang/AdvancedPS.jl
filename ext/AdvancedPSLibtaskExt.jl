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

"""
    LibtaskModel{F}

State wrapper to hold `Libtask.CTask` model initiated from `f`.
"""
function AdvancedPS.LibtaskModel(
    f::AdvancedPS.AbstractGenericModel, rng::Random.AbstractRNG, args...
) # Changed the API, need to take care of the RNG properly
    return AdvancedPS.LibtaskModel(
        f,
        Libtask.TapedTask(
            f, rng, args...; deepcopy_types=Union{AdvancedPS.TracedRNG,typeof(f)}
        ),
    )
end

function Base.copy(model::AdvancedPS.LibtaskModel)
    return AdvancedPS.LibtaskModel(model.f, copy(model.ctask))
end

const LibtaskTrace{R} = AdvancedPS.Trace{<:AdvancedPS.LibtaskModel,R}

function AdvancedPS.Trace(
    model::AdvancedPS.AbstractGenericModel, rng::Random.AbstractRNG, args...
)
    return AdvancedPS.Trace(AdvancedPS.LibtaskModel(model, rng, args...), rng)
end

# step to the next observe statement and
# return the log probability of the transition (or nothing if done)
function AdvancedPS.advance!(t::LibtaskTrace, isref::Bool=false)
    isref ? AdvancedPS.load_state!(t.rng) : AdvancedPS.save_state!(t.rng)
    AdvancedPS.inc_counter!(t.rng)

    # Move to next step
    return Libtask.consume(t.model.ctask)
end

# create a backward reference in task_local_storage
function AdvancedPS.addreference!(task::Task, trace::LibtaskTrace)
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:__trace] = trace

    return task
end

function AdvancedPS.update_rng!(trace::LibtaskTrace)
    rng, = trace.model.ctask.args
    trace.rng = rng
    return trace
end

# Task copying version of fork for Trace.
function AdvancedPS.fork(trace::LibtaskTrace, isref::Bool=false)
    newtrace = copy(trace)
    AdvancedPS.update_rng!(newtrace)
    isref && AdvancedPS.delete_retained!(newtrace.model.f)
    isref && delete_seeds!(newtrace)

    # add backward reference
    AdvancedPS.addreference!(newtrace.model.ctask.task, newtrace)
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function AdvancedPS.forkr(trace::LibtaskTrace)
    newf = AdvancedPS.reset_model(trace.model.f)
    Random123.set_counter!(trace.rng, 1)

    ctask = Libtask.TapedTask(
        newf, trace.rng; deepcopy_types=Union{AdvancedPS.TracedRNG,typeof(trace.model.f)}
    )
    new_tapedmodel = AdvancedPS.LibtaskModel(newf, ctask)

    # add backward reference
    newtrace = AdvancedPS.Trace(new_tapedmodel, trace.rng)
    AdvancedPS.addreference!(ctask.task, newtrace)
    AdvancedPS.gen_refseed!(newtrace)
    return newtrace
end

AdvancedPS.update_ref!(::LibtaskTrace) = nothing

"""
    observe(dist::Distribution, x)

Observe sample `x` from distribution `dist` and yield its log-likelihood value.
"""
function AdvancedPS.observe(dist::Distributions.Distribution, x)
    return Libtask.produce(Distributions.loglikelihood(dist, x))
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
            AdvancedPS.addreference!(trace.model.ctask.task, trace) # TODO: Do we need it here ?
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
    return AdvancedPS.PGSample(replayed.model.f, logevidence),
    AdvancedPS.PGState(newtrajectory)
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
        AdvancedPS.addreference!(trace.model.ctask.task, trace) # Do we need it here ?
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
