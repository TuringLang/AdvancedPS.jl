# Idea: We decouple tasks form the model by only allowing to pass a function.
# This way, we do no longer need to have the model and the sampler saved in the trace struct
# However, we still need to insantiate the VarInfo

# TaskInfo stores additional information about the task
mutable struct Trace{Tvi, TInfo}  <: AbstractTrace where {Tvi, TInfo <: AbstractTaskInfo}
    vi::Tvi # Unfortunatley, we can not set force this to be a subtype of VarInfo...
    task::Task
    taskinfo::TInfo
end


# NOTE: this function is called by `forkr`

function Trace(vi::Tvi, f::Function, taskinfo::TInfo) where {Tvi,TInfo<:AbstractTaskInfo}
    # CTask(()->f());
    task = CTask( () -> begin res=f(); produce(Val{:done}); res; end )
    res = Trace(copy(vi), task, copy(taskinfo))
    if res.task.storage === nothing
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    return res
end

function Base.copy(trace::Trace{Tvi,TInfo}) where {Tvi, TInfo <: AbstractTaskInfo}
    return get_new_trace(trace.vi, trace.task, trace.taskinfo)
end


# A bit an ugly solution...
function get_new_trace(vi::Tvi, task::Task, taskinfo::TInfo) where {Tvi,TInfo<:AbstractTaskInfo}
    res = Trace(copy(vi), Libtask.copy(task), copy(taskinfo))
    # CTask(()->f());
    if res.task.storage === nothing
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage

    return res
end



# step to the next observe statement, return log likelihood
Libtask.consume(t::Trace) = (t.vi.num_produce += 1; consume(t.task))

# Task copying version of fork for Trace.
function fork(trace::Trace, is_ref::Bool = false, set_retained_vns_del_by_spl!::Union{Function,Nothing} = nothing)
    newtrace = copy(trace)
    if is_ref
        @assert set_retained_vns_del_by_spl! !== nothing "[AdvancedPS] set_retained_vns_del_by_spl! is not set."
        set_retained_vns_del_by_spl!(newtrace.vi)
    end
    newtrace.task.storage[:turing_trace] = newtrace
    return newtrace
end


function forkr(trace::Trace)
    newtrace = Trace(trace.vi, trace.task.code, trace.taskinfo)
    reset_task_info!(trace.taskinfo)
    newtrace.vi.num_produce = 0
    return newtrace
end

current_trace() = current_task().storage[:turing_trace]

const Particle = Trace
