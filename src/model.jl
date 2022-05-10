"""
    GenericModel{F}

State wrapper to hold `Libtask.CTask` model initiated from `f`
"""
struct GenericModel{F} <: AbstractMCMC.AbstractModel
    f::F
    ctask::Libtask.TapedTask{F}
end

GenericModel(f, args...) = GenericModel(f, Libtask.TapedTask(f, args...))
Base.copy(model::GenericModel) = GenericModel(model.f, copy(model.ctask))

"""
    Trace{F,R}
"""
mutable struct Trace{F,R}
    model::F
    rng::R

    Trace(model::F, rng::R) where {F,R} = new{F,R}(model, rng) # Bring back the default since we override

    function Trace(model::F, rng::R) where {F<:GenericModel,R}
        # add backward reference
        newtrace = new{F,R}(model, rng)
        addreference!(model.ctask.task, newtrace)
        return newtrace
    end
end

const Particle = Trace

const SSMTrace{R} = Trace{<:AbstractStateSpaceModel,R}
const GenericTrace{R} = Trace{<:GenericModel,R}

# Copy task
Base.copy(trace::GenericTrace) = Trace(copy(trace.model), deepcopy(trace.rng))

# create a backward reference in task_local_storage
function addreference!(task::Task, trace::GenericTrace)
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:__trace] = trace

    return task
end

current_trace() = current_task().storage[:__trace]

# Task copying version of fork for Trace.
function fork(trace::GenericTrace, isref::Bool=false)
    newtrace = copy(trace)
    rng, = newtrace.model.ctask.args
    newtrace.rng = rng
    isref && delete_retained!(newtrace.model.f)
    isref && delete_seeds!(newtrace)

    # add backward reference
    addreference!(newtrace.model.ctask.task, newtrace)
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace::GenericTrace)
    newf = reset_model(trace.model.f)
    Random123.set_counter!(trace.rng, 1)

    ctask = Libtask.TapedTask(newf, trace.rng)
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

function delete_seeds!(particle::Particle)
    return particle.rng.keys = particle.rng.keys[1:(particle.rng.count - 1)]
end
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
