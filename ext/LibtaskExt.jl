module LibtaskExt

using AdvancedPS: AdvancedPS
using Random123: Random123
using Distributions: Distributions

isdefined(Base, :get_extension) ? (
using Libtask: Libtask) : (
using ..Libtask: Libtask)

"""
    GenericModel{F}

State wrapper to hold `Libtask.CTask` model initiated from `f`
"""
struct GenericModel{F1,F2} <: AbstractMCMC.AbstractModel
    f::F1
    ctask::Libtask.TapedTask{F2}

    GenericModel(f::F1, ctask::Libtask.TapedTask{F2}) where {F1,F2} = new{F1,F2}(f, ctask)
end

function GenericModel(f, args...)
    return GenericModel(
        f, Libtask.TapedTask(f, args...; deepcopy_types=Union{TracedRNG,typeof(f)})
    )
end
Base.copy(model::GenericModel) = GenericModel(model.f, copy(model.ctask))

mutable struct LibtaskTrace{F,R}
    model::F
    rng::R

    function LibtaskTrace(model::F, rng::R) where {F<:GenericModel,R}
        # add backward reference
        newtrace = new{F,R}(model, rng)
        addreference!(model.ctask.task, newtrace)
        return newtrace
    end
end

# step to the next observe statement and
# return the log probability of the transition (or nothing if done)
function AdvancedPS.advance!(t::LibtaskTrace, isref::Bool=false)
    isref ? load_state!(t.rng) : save_state!(t.rng)
    inc_counter!(t.rng)

    # Move to next step
    return Libtask.consume(t.model.ctask)
end

# Copy task
Base.copy(trace::LibtaskTrace) = LibtaskTrace(copy(trace.model), deepcopy(trace.rng))

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
    newf = reset_model(trace.model.f)
    Random123.set_counter!(trace.rng, 1)

    ctask = Libtask.TapedTask(
        newf, trace.rng; deepcopy_types=Union{TracedRNG,typeof(trace.model.f)}
    )
    #ctask = Libtask.TapedTask(newf, trace.rng)
    new_tapedmodel = GenericModel(newf, ctask)

    # add backward reference
    newtrace = Trace(new_tapedmodel, trace.rng)
    addreference!(ctask.task, newtrace)
    gen_refseed!(newtrace)
    return newtrace
end

function gen_refseed!(part::Particle)
    seed = split(state(part.rng.rng), 1)
    return safe_set_refseed!(part.rng, seed[1])
end

# reset log probability
reset_logprob!(::Particle) = nothing

reset_model(f) = deepcopy(f)
delete_retained!(f) = nothing

"""
    observe(dist::Distribution, x)

Observe sample `x` from distribution `dist` and yield its log-likelihood value.
"""
function observe(dist::Distributions.Distribution, x)
    return Libtask.produce(Distributions.loglikelihood(dist, x))
end

"""
    delete_seeds!(particle::Particle)

Truncate the seed history from the `particle` rng. When forking the reference Particle
we need to keep the seeds up to the current model iteration but generate new seeds
and random values afterward.
"""
function delete_seeds!(particle::Particle)
    return particle.rng.keys = particle.rng.keys[1:(particle.rng.count - 1)]
end

end
