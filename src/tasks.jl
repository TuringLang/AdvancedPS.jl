# Idea: We decouple tasks form the model by only allowing to pass a function.
# This way, we do no longer need to have the model and the sampler saved in the trace struct
# However, we still need to insantiate the VarInfo

# TaskInfo stores additional informatino about the task
mutable struct Trace{Tvi, TInfo}  <: AbstractTrace where {Tvi, TInfo <: AbstractTaskInfo}
    vi::Tvi # Unfortunatley, we can not set force this to be a subtype of VarInfo...
    task::Task
    taskinfo::TInfo
end

function Base.copy(trace::Trace{Tvi,TInfo}, copy_vi::Function) where {Tvi, TInfo <: AbstractTaskInfo}
    res = Trace{typeof(trace.vi),typeof(trace.taskinfo)}(copy_vi(trace.vi), copy(trace.task), copy(trace.taskinfo))
    return res
end

# The procedure passes a function which is specified by the model.

function Trace(f::Function, vi, taskinfo, copy_vi::Function)
    task = CTask( () -> begin res=f(); produce(Val{:done}); res; end )

    res = Trace{typeof(vi),typeof(task)}(copy_vi(vi), task, copy(taskinfo));
    # CTask(()->f());
    if res.task.storage === nothing
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    return res
end

## We need to design the task in the Turing wrapper.
function Trace( vi, task::Task, taskinfo, copy_vi::Function)


    res = Trace{typeof(vi),typeof(taskinfo)}(copy_vi(vi), Libtask.copy(task), copy(taskinfo));
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
function fork(trace :: Trace, copy_vi::Function, is_ref :: Bool = false, set_retained_vns_del_by_spl!::Union{Function,Nothing} = nothing)
    newtrace = copy(trace,copy_vi)
    if is_ref
        @assert set_retained_vns_del_by_spl! != nothing "[AdvancedPF] set_retained_vns_del_by_spl! is not set."
        set_retained_vns_del_by_spl!(newtrace.vi)
    end
    newtrace.task.storage[:turing_trace] = newtrace
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace :: Trace, copy_vi::Function)

    newtrace = Trace(trace.vi,trace.task ,trace.taskinfo,copy_vi)
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
mutable struct ParticleContainer{Tvi,TInfo} <: AbstractParticleContainer where {Tvi, TInfo <: AbstractTaskInfo}
    vals::Vector{Trace{Tvi,TInfo}}
    # A named tuple with functions to manipulate the particles vi.
    manipulators::Dict{String,Function}
    # logarithmic weights (Trace) or incremental log-likelihoods (ParticleContainer)
    logWs::Vector{Float64}
    #This corresponds to log p(x_{0:t}), therefore the log likelihood of the transitions up to this point.
    #Especially important for ancestor sampling.
    logpseq::Vector{Float64}
    # log model evidence
    logE::Float64
    # helpful for rejuvenation steps, e.g. in SMC2
    n_consume::Int


end

## Empty initilaizaiton
ParticleContainer{Tvi,TInfo}() where {Tvi, TInfo <: AbstractTaskInfo} = ParticleContainer{Tvi,TInfo}(Dict{String,Function}(),0)
ParticleContainer{Tvi,TInfo}(manipulators::Dict{String,Function}) where {Tvi, TInfo <: AbstractTaskInfo} = ParticleContainer{Tvi,TInfo}(manipulators,0)
#Some initilaizaiton...
function ParticleContainer{Tvi,TInfo}(manipulators::Dict{String,Function}, n::Int) where {Tvi, TInfo <: AbstractTaskInfo}
    if !("copy" in keys(manipulators)) manipulators["copy"] = deepcopy end
    if !("set_retained_vns_del_by_spl!" in keys(manipulators)) manipulators["set_retained_vns_del_by_spl!"] = (x) -> () end
    ParticleContainer{Tvi,TInfo}( Vector{Trace{Tvi,TInfo}}(undef, n), manipulators, Float64[],Float64[], 0.0, 0)

end


Base.collect(pc :: ParticleContainer) = pc.vals # prev: Dict, now: Array
Base.length(pc :: ParticleContainer)  = length(pc.vals)
Base.similar(pc :: ParticleContainer{T}) where T = ParticleContainer{T}(0)
# pc[i] returns the i'th particle
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]



# registers a new x-particle in the container
function Base.push!(pc::ParticleContainer, p::Particle)
    push!(pc.vals, p)
    push!(pc.logWs, 0.0)
    push!(pc.logpseq,0.0)
    pc
end

function extend!(pc::ParticleContainer, n::Int, varInfo, tasks::Task, taskInfo::T) where T <: AbstractTaskInfo
    # compute total number of particles number of particles
    n0 = length(pc)
    ntotal = n0 + n

    # add additional particles and weights
    vals = pc.vals
    logWs = pc.logWs
    logpseq = pc.logpseq
    resize!(vals, ntotal)
    resize!(logWs, ntotal)
    resize!(logpseq, ntotal)

    @inbounds for i in (n0 + 1):ntotal
        vals[i] = Trace(varInfo,tasks, taskInfo, pc.manipulators["copy"])
        logWs[i] = 0.0
        logpseq[i] = 0.0
    end

    pc
end
# clears the container but keep params, logweight etc.
function Base.empty!(pc::ParticleContainer)
    pc.vals  = eltype(pc.vals)[]
    pc.logWs = Float64[]
    pc.logpseq = Float64[]
    pc
end

# clones a theta-particle
function Base.copy(pc::ParticleContainer)
    # fork particles
    vals = eltype(pc.vals)[fork(p, pc.mainpulators["copy"]) for p in pc.vals]

    # copy weights
    logWs = copy(pc.logWs)
    logpseq = copy(pc.logpseq)

    ParticleContainer( vals, pc.manipulators, logWs, logpseq, pc.logE, pc.n_consume)
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

        if in(:logpseq,fieldnames(typeof(p.taskinfo))) && isa(p.taskinfo.logpseq,Float64)
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
            println(score)
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
weights(pc::ParticleContainer) = softmax!(copy(pc.logWs))

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
    (pc.logpseq[t] = logp)

increase_logevidence(pc :: ParticleContainer, logw :: Float64) =
    (pc.logE += logw)
