"""
    TapedModel{F}

State wrapper to hold `Libtask.CTask` model initiated from `f`
"""
struct TapedModel{F} <: AbstractMCMC.AbstractModel
    f::F
    ctask::Libtask.TapedTask{F}
end

TapedModel(f, args...) = TapedModel(f, Libtask.TapedTask(f, args...))
Base.copy(model::TapedModel) = TapedModel(model.f, copy(model.ctask))

"""
    Trace{F,R}
"""
struct Trace{F,R}
    model::F
    rng::R

    Trace(model::F, rng::R) where {F,R} = new{F,R}(model, rng) # Bring back the default since we override

    function Trace(model::F, rng::R) where {F<:TapedModel,R}
        # add backward reference
        newtrace = new{F,R}(model, rng)
        addreference!(model.ctask.task, newtrace)
        return newtrace
    end
end

const Particle = Trace

const SSMTrace{R} = Trace{<:AbstractStateSpaceModel,R}
const TapedTrace{R} = Trace{<:TapedModel,R}

# Copy task
Base.copy(trace::TapedTrace) = Trace(copy(trace.model), deepcopy(trace.rng))

# create a backward reference in task_local_storage
function addreference!(task::Task, trace::TapedTrace)
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:__trace] = trace

    return task
end

current_trace() = current_task().storage[:__trace]

# Task copying version of fork for Trace.
function fork(trace::TapedTrace, isref::Bool=false)
    newtrace = copy(trace)
    isref && delete_retained!(newtrace.model.f)

    # add backward reference
    addreference!(newtrace.model.ctask.task, newtrace)
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace::TapedTrace)
    newf = reset_model(trace.model.f)
    Random123.set_counter!(trace.rng, 1)

    ctask = Libtask.TapedTask(newf, trace.rng)
    new_tapedmodel = TapedModel(newf, ctask)

    # add backward reference
    newtrace = Trace(new_tapedmodel, trace.rng)
    addreference!(ctask.task, newtrace)
    gen_refseed!(newtrace)
    return newtrace
end

function gen_refseed!(part::Particle)
    seed = split(state(part.rng.rng), 1)
    return set_refseed!(part.rng, seed[1])
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
