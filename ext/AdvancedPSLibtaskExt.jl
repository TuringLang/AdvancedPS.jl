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

State wrapper to hold `Libtask.CTask` model initiated from `f`
"""
struct LibtaskModel{F1,F2}
    f::F1
    ctask::Libtask.TapedTask{F2}

    LibtaskModel(f::F1, ctask::Libtask.TapedTask{F2}) where {F1,F2} = new{F1,F2}(f, ctask)
end

function LibtaskModel(f, args...)
    return LibtaskModel(
        f,
        Libtask.TapedTask(f, args...; deepcopy_types=Union{AdvancedPS.TracedRNG,typeof(f)}),
    )
end

Base.copy(model::LibtaskModel) = LibtaskModel(model.f, copy(model.ctask))

const LibtaskTrace{R} = AdvancedPS.Trace{<:LibtaskModel,R}

# step to the next observe statement and
# return the log probability of the transition (or nothing if done)
function AdvancedPS.advance!(t::LibtaskTrace, isref::Bool=false)
    isref ? AdvancedPS.load_state!(t.rng) : AdvancedPS.save_state!(t.rng)
    AdvancedPS.inc_counter!(t.rng)

    # Move to next step
    return Libtask.consume(t.model.ctask)
end

# create a backward reference in task_local_storage
function addreference!(task::Task, trace::LibtaskTrace)
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:__trace] = trace

    return task
end

current_trace() = current_task().storage[:__trace]

function update_rng!(trace::LibtaskTrace)
    rng, = trace.model.ctask.args
    trace.rng = rng
    return trace
end

# Task copying version of fork for Trace.
function AdvancedPS.fork(trace::LibtaskTrace, isref::Bool=false)
    newtrace = copy(trace)
    update_rng!(newtrace)
    isref && delete_retained!(newtrace.model.f)
    isref && delete_seeds!(newtrace)

    # add backward reference
    addreference!(newtrace.model.ctask.task, newtrace)
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
    new_tapedmodel = LibtaskModel(newf, ctask)

    # add backward reference
    newtrace = AdvancedPS.Trace(new_tapedmodel, trace.rng)
    addreference!(ctask.task, newtrace)
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
    model::AbstractMCMC.AbstractModel,
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
            gen_model = LibtaskModel(deepcopy(model), trng)
            trace = AdvancedPS.Trace(LibtaskModel(deepcopy(model), trng), trng)
            addreference!(gen_model.ctask.task, trace) # Do we need it here ?
            trace
        end
    end

    particles = AdvancedPS.ParticleContainer(traces, AdvancedPS.TracedRNG(), rng)

    # Perform a particle sweep.
    reference = isref ? particles.vals[nparticles] : nothing
    logevidence = AdvancedPS.sweep!(rng, particles, sampler.resampler, reference)

    # Pick a particle to be retained.
    newtrajectory = rand(rng, particles)

    return AdvancedPS.PGSample(newtrajectory, logevidence),
    AdvancedPS.PGState(newtrajectory)
end

function AbstractMCMC.sample(
    model::AbstractMCMC.AbstractModel, sampler::AdvancedPS.SMC; kwargs...
)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, sampler; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::AdvancedPS.SMC;
    kwargs...,
)
    if !isempty(kwargs)
        @warn "keyword arguments $(keys(kwargs)) are not supported by `SMC`"
    end

    traces = map(1:(sampler.nparticles)) do i
        trng = AdvancedPS.TracedRNG()
        gen_model = LibtaskModel(deepcopy(model), trng)
        trace = AdvancedPS.Trace(LibtaskModel(deepcopy(model), trng), trng)
        addreference!(gen_model.ctask.task, trace) # Do we need it here ?
        trace
    end

    # Create a set of particles.
    particles = AdvancedPS.ParticleContainer(traces, AdvancedPS.TracedRNG(), rng)

    # Perform particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, sampler.resampler)

    return AdvancedPS.SMCSample(
        collect(particles), AdvancedPS.getweights(particles), logevidence
    )
end

end
