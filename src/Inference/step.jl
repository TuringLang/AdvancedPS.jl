
#SMC step
function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractPFModel,
    spl::SPL,
    ::Integer;
    iteration::Integer,
    kwargs...
    ) where SPL <: SMCSampler

    particle = spl.pc[iteration]

    params = spl.uf.tonamedtuple(particle.vi)
    return PFTransition(params, particle.taskinfo.logp, spl.pc.logE, weights(spl.pc)[iteration])
end


# PG step
function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractPFModel,
    spl::SPL,
    ::Integer;
    kwargs...
    ) where SPL <: PGSampler

    n = spl.alg.n

    T = Trace{typeof(spl.vi), PGTaskInfo{Float64}}

    if spl.ref_traj !== nothing
        particles = T[ get_new_trace(spl.vi, model.task, PGTaskInfo()) for _ =1:n-1]
        pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n-1),0.0,0)
        # Reset Task
        spl.ref_traj = forkr(spl.ref_traj)
        push!(pc, spl.ref_traj)
    else
        particles = T[ get_new_trace(spl.vi, model.task, PGTaskInfo()) for _ =1:n]
        pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n),0.0,0)
    end

    sample!(pc, spl.alg, spl.uf, spl.ref_traj)

    indx = AdvancedPS.randcat(weights(pc))
    particle = spl.ref_traj = pc[indx]
    params = spl.uf.tonamedtuple(particle.vi)
    return PFTransition(params, particle.taskinfo.logp, pc.logE, weights(pc)[indx])
end


#
# # The resampler threshold is only imprtant for the first step!
# function samplePGAS!(pc::ParticleContainer,resampler::Function = resample_systematic ,ref_particle::Union{Particle,Nothing}=nothing ,joint_logp::Union{Tuple{Vector{Symbol},Function},Nothing}=nothing, resampler_threshold =0.5)
#     if ref_particle === nothing
#         # We do not have a reference trajectory yet, therefore, perform normal SMC!
#         sampleSMC!(pc,resampler,resampler_threshold)
#
#     else
#         # Before starting, we need to copute the ancestor weights.
#         # Note that there is a inconsistency with the original paper
#         # http://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf
#         # Lindsten samples already for x1. This is not possible in
#         # in this situation because we do not have access to the information
#         # which variables belong to x1 and which to x0!
#
#         # The procedure works as follows. The ancestor weights are only dependent on
#         # the states x_{0:t-1}. We make us of this by computing the ancestor indices for
#         # the next state.
#         ancestor_index = length(pc)
#         ancestor_particle::Union{typeof(pc[1]), Nothing} = nothing
#         n = length(pc)
#
#         num_totla_consume = typemax(Int64)
#         first_round = true
#         while consume(pc) != Val{:done}
#             # compute weights
#             Ws = weights(pc)
#             # We need them for ancestor sampling...
#             logws = copy(pc.logWs)
#             logpseq = [pc[i].taskinfo.logpseq for i in 1:n ]
#             # check that weights are not NaN
#             @assert !any(isnan, Ws)
#             # sample ancestor indices
#             # Ancestor trajectory is not sampled
#             nresamples = n-1
#             indx = resampler(Ws, nresamples)
#
#             # Now the ancestor sampling starts. We do not use ancestor sampling in the
#             # first step. This is due to the reason before. In addition, we do not need
#             # to compute the ancestor index for the last step, because we are always
#             #computing the ancestor index one step ahead.
#             if ancestor_particle === nothing
#                 push!(indx,n)
#                 resample!(pc, indx,ref_particle)
#                 # In this case, we do have access to the joint_logp ! Therefore:
#                 if joint_logp !== nothing
#                     # We need to create a dictionary with symbol value paris, which we pass
#                     # to the joint_logp function.
#                     @assert  isa(joint_logp[1], Vector{Symbols}) "[AdvancedPS] the first argument of the joint_logp tuble must be a vector of Symbols!"
#                     @assert  isa(joint_logp[2], Function) "[AdvancedPS] the second argument of the joint_logp tuble must be a function of the for"*
#                                                                 " f(num_produce, args...), which returns a value of Float64!"
#
#                     # We are executing the joint_logp function as a set of tasks. The idea behind
#                     # this is that we might extend the task to include parallelization.
#                     tasks = []
#                     for i = 1:n
#                         args = pc.manipulators["get_AS_joint_logp_args"](joint_logp[1],pc[i],ref_particle)
#                         task = CTask( () -> begin result=joint_logp[2](pc.n_consume,args...); produce(result); end )
#                         push(tasks,task)
#                         schedule(task)
#                         yield()
#                     end
#                     for (i,t) in enumerate(tasks)
#                         logws[i] += consume(t) -logpseq[i]  # The ancestor weights w_ancstor = w_i *p(x_{0:t-1},x_{t:T})/p(x_{0:t-1})
#                     end
#
#                     ancestor_index = randcat(softmax!(logws))
#                     # We are one step behind....
#                     selected_path = pc[ancestor_index]
#                     new_vi = pc.manipulators["merge_traj"](pc.manipulators["copy"](selected_path.vi),ref_particle.vi,pc.n_consume +1)
#                     ancestor_particle = Trace{typeof(new_vi),typeof(selected_path.taskinfo)}(new_vi,copy(selected_path.task),copy(selected_path.taskinfo))
#                 else
#                     if pc.n_consume <= num_totla_consume-1 #We do not need to sample the last one...
#                         # The idea is rather simple, we extend the vs and let them run trough...
#                         pc_ancestor = ParticleContainer{typeof(pc[1].vi),typeof(pc[1].taskinfo)}()
#                         pc_ancestor.n_consume = pc.n_consume
#                         for i = 1:n
#                             new_vi = pc.manipulators["merge_traj"](pc.manipulators["copy"](pc[i].vi),ref_particle.vi)
#                             new_particle = Trace{typeof(new_vi),typeof(pc[i].taskinfo)}(new_vi,copy(pc[i].task),copy(pc[i].taskinfo))
#                             push!(pc_ancestor,new_particle)
#                         end
#
#                         while consume(pc_ancestor) != Val{:done} end # No resampling, we just want to get log p(x_{0:t-1},x'_{t,T})
#                         for i in 1:n
#                             logws[i] += pc_ancestor[i].taskinfo.logpseq -logpseq[i]  # The ancestor weights w_ancstor = w_i *p(x_{0:t-1},x_{t:T})/p(x_{0:t-1})
#                         end
#
#
#                         ancestor_index = randcat(softmax!(logws))
#                         # We are one step behind....
#                         selected_path = pc[ancestor_index]
#                         new_vi = pc.manipulators["merge_traj"](pc.manipulators["copy"](selected_path.vi),ref_particle.vi,pc.n_consume +1)
#                         ancestor_particle = Trace{typeof(new_vi),typeof(selected_path.taskinfo)}(new_vi,copy(selected_path.task),copy(selected_path.taskinfo))
#                         num_total_consume = pc_ancestor.n_consume
#                     end
#                 end
#             end
#             # Reactivate the tasks, this is only important for synchorinty.
#             for i =1:n
#                 consume(pc[i].task)
#             end
#         end
#     end
#
# end
