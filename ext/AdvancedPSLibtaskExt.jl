module AdvancedPSLibtaskExt

if isdefined(Base, :get_extension)
    using AdvancedPS: AdvancedPS
    using AdvancedPS: Random123
    using AdvancedPS: AbstractMCMC
    using Libtask: Libtask
else
    using ..AdvancedPS: AdvancedPS
    using ..AdvancedPS: Random123
    using ..AdvancedPS: AbstractMCMC
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

function AdvancedPS.Trace(model::AbstractMCMC.AbstractModel, rng::AdvancedPS.TracedRNG)
    gen_model = LibtaskModel(model, rng)
    trace = AdvancedPS.Trace(gen_model, rng)
    addreference!(gen_model.ctask.task, trace) # Do we need it here ?
    return trace
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

end
