

using Turing.Core.RandomVariables
using Turing



# Idea: We decouple tasks form the model by only allowing to pass a function.
# This way, we do no longer need to have the model and the sampler saved in the trace struct
# However, we still need to insantiate the VarInfo

# TaskInfo stores additional informatino about the task
mutable struct Trace{Tvi, Tinfo <:AbstractTaskInfo}
    vi::Tvi # Unfortunatley, we can not set force this to be a subtype of VarInfo...
    task::Task
    taksinfo::Tinfo
    function Trace{AbstractTaskInfo}(vi, taskinfo<:AbstractTaskInfo, task::Task)
        return new{typeof(vi),typeof(taskinfo)}(vi, task, taskinfo)
    end
end

function Base.copy(trace::Trace, copy_vi::Function)
    try
        vi = copy_vi(trace.vi)
    catch e
        error("[AdvancedPS] The copy function given by the manipulators must only accept vi as argument")
    end
    taskinfo = copy(trace.taskinfo)
    res = Trace{AbstractTaskInfo}(vi, copy(trace.task), taskinfo)
    return res
end

# The procedure passes a function which is specified by the model.

function Trace(f::Function, vi<:RandomVariables.AbstractVarInfo, taskinfo<:AbstractTaskInfo, copy_vi::Function) where {T <: AbstractSampler}
    task = CTask( () -> begin res=f(); produce(Val{:done}); res; end )

    res = Trace{AbstractTaskInfo}(copy_vi(vi), task, copy(taskinfo));
    # CTask(()->f());
    if res.task.storage === nothing
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    return res
end

## We need to design the task in the Turing wrapper.
function Trace(task::Task, vi<:RandomVariables.AbstractVarInfo, taskinfo<:AbstractTaskInfo, copy_vi::Function)
    try
        new_vi = copy_vi(vi)
    catch e
        error("[AdvancedPS] The copy function given by the manipulators must only accept vi as argument")
    end
    res = Trace{AbstractTaskInfo}(new_vi, task, copy(taskinfo));
    # CTask(()->f());
    res.task = task
    if res.task.storage === nothing
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    return res
end



# step to the next observe statement, return log likelihood
Libtask.consume(t::Trace) = (t.vi.num_produce += 1; consume(t.task))

# Task copying version of fork for Trace.
function fork(trace :: Trace, copy_vi::Function, is_ref :: Bool = false, set_retained_vns_del_by_spl!::Function = nothing)
    newtrace = copy(trace)
    if is_ref
        @assert set_retained_vns_del_by_spl! != nothing "[AdvancedPF] set_retained_vns_del_by_spl! is not set."
        try
            set_retained_vns_del_by_spl!(newtrace.vi, taskinfo.idcs)
        catch e
            error("[AdvancedPS] set_retained_vns_del_by_spl! must have the form set_retained_vns_del_by_spl!(newtrace.vi, taskinfo.idcs)")
        end
    newtrace.task.storage[:turing_trace] = newtrace
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace :: Trace, copy_vi::Function)
    try
        new_vi = copy_vi(trace.vi)
    catch e
        error("[AdvancedPS] The copy function given by the manipulators must only accept vi as argument")
    end
    newtrace = Trace(trace.task.code,new_vi ,copy(trace.taskinfo))
    newtrace.vi.num_produce = 0
    return newtrace
end

current_trace() = current_task().storage[:turing_trace]

const Particle = Trace

"""
Data structure for particle filters
- effectiveSampleSize(pc :: ParticleContainer)
- normalise!(pc::ParticleContainer)
- consume(pc::ParticleContainer): return incremental likelihood
"""
mutable struct ParticleContainer{T<:Particle, F}
    vals::Vector{T}
    # A named tuple with functions to manipulate the particles vi.
    manipulators::NamedTuple{}
    # logarithmic weights (Trace) or incremental log-likelihoods (ParticleContainer)
    logWs::Vector{Float64}
    #This corresponds to log p(x_{0:t}), therefore the log likelihood of the transitions up to this point.
    #Especially important for ancestor sampling.
    logPseq::Vector{Float64}
    # log model evidence
    logE::Float64
    # helpful for rejuvenation steps, e.g. in SMC2
    n_consume::Int
end

## Empty initilaizaiton
ParticleContainer{T}() where T = ParticleContainer{T}(NamedTuple{}(),0)
ParticleContainer{T}(manipulators::NamedTuple{}) where T = ParticleContainer{T}(manipulators,0)


#Some initilaizaiton...
function ParticleContainer{T}(manipulators::NamedTuple{}, n::Int) where {T}
    if !(:copy in keys(pc.manipulators)) manipulators = merge(manipulators, (copy = deepcopy,) end
    if !(:set_retained_vns_del_by_spl! in keys(pc.manipulators)) manipulators = merge(manipulators, (set_retained_vns_del_by_spl! = (args) -> (),)) end
    ParticleContainer( Vector{T}(undef, n), manipulators, Float64[],Float64[], 0.0, 0)

end


Base.collect(pc :: ParticleContainer) = pc.vals # prev: Dict, now: Array
Base.length(pc :: ParticleContainer)  = length(pc.vals)
Base.similar(pc :: ParticleContainer{T}) where T = ParticleContainer{T}(0)
# pc[i] returns the i'th particle
Base.getindex(pc :: ParticleContainer, i :: Real) = pc.vals[i]


# registers a new x-particle in the container
function Base.push!(pc::ParticleContainer, p::Particle)
    push!(pc.vals, p)
    push!(pc.logWs, 0.0)
    push!(pc.logPseq,0.0)
    pc
end

function Base.push!(pc::ParticleContainer, n::Int, varInfo::VarInfo, tasks::Task, taskInfo<:AbstractTaskInfo)
    # compute total number of particles number of particles
    n0 = length(pc)
    ntotal = n0 + n

    # add additional particles and weights
    vals = pc.vals
    logWs = pc.logWs
    model = pc.model
    resize!(vals, ntotal)
    resize!(logWs, ntotal)
    resize!(logPseq, ntotal)

    @inbounds for i in (n0 + 1):ntotal
        vals[i] = Trace(varInfo, tasks, taskInfo)
        logWs[i] = 0.0
        logPseq[i] = 0.0
    end

    pc
end

# clears the container but keep params, logweight etc.
function Base.empty!(pc::ParticleContainer)
    pc.vals  = eltype(pc.vals)[]
    pc.logWs = Float64[]
    pc.logPseq = Float64[]
    pc
end

# clones a theta-particle
function Base.copy(pc::ParticleContainer)
    # fork particles
    vals = eltype(pc.vals)[fork(p, pc.mainpulators["copy"]) for p in pc.vals]

    # copy weights
    logWs = copy(pc.logWs)
    logPseq = copy(pc.logPseq)

    ParticleContainer(pc.model, vals, logWs, logPseq, pc.logE, pc.n_consume)
end

# run particle filter for one step, return incremental likelihood
function Libtask.consume(pc :: ParticleContainer)
    # normalisation factor: 1/N
    z1 = logZ(pc)
    n = length(pc)

    particles = collect(pc)
    num_done = 0
    for i=1:n
        p = particles[i]
        score = Libtask.consume(p)

        if in(:logpseq,fieldnames(p.taskinfo)) && isa(p.taskinfo.logpseq,Float64)
            set_logpseg(pc,i,p.taskinfo.logpseq)
        end

        if score isa Real
            score += p.taskinfo.logp

            ## Equivalent to reset logp
            p.taskinfo.logp = 0

            increase_logweight(pc, i, Float64(score))
        elseif score == Val{:done}
            num_done += 1
        else
            error("[consume]: error in running particle filter.")
        end
    end

    if num_done == n
        res = Val{:done}
    elseif num_done != 0
        error("[consume]: mis-aligned execution traces, num_particles= $(n), num_done=$(num_done).")
    else
        # update incremental likelihoods
        z2 = logZ(pc)
        res = increase_logevidence(pc, z2 - z1)
        pc.n_consume += 1
        # res = increase_loglikelihood(pc, z2 - z1)
    end

    res
end

# compute the normalized weights
weights(pc::ParticleContainer) = softmax(pc.logWs)

# compute the log-likelihood estimate, ignoring constant term ``- \log num_particles``
logZ(pc::ParticleContainer) = logsumexp(pc.logWs)

# compute the effective sample size ``1 / ∑ wᵢ²``, where ``wᵢ```are the normalized weights
function effectiveSampleSize(pc :: ParticleContainer)
    Ws = weights(pc)
    return inv(sum(abs2, Ws))
end

increase_logweight(pc :: ParticleContainer, t :: Int, logw :: Float64) =
    (pc.logWs[t]  += logw)


set_logpseg(pc :: ParticleContainer, t :: Int, logp :: Float64) =
    (pc.logPseq[t] = logp)

increase_logevidence(pc :: ParticleContainer, logw :: Float64) =
    (pc.logE += logw)


    ###

    ### Resample Steps

    ###
