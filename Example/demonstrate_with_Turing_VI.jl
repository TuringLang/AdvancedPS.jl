
## It is not yet a package...
using Distributions
using Turing.Core.RandomVariables
import Turing.Core:  @varname
import Turing.Utilities: vectorize
using Turing
using Revise
using AdvancedPS
import Turing.Core: tonamedtuple
include("AdvancedPS/Example/turing_interface.jl")
# Define a short model.
# The syntax is rather simple. Observations need to be reported with report_observation.
# Transitions must be reported using report_transition.
# The trace contains the variables which we want to infer using particle gibbs.
# Thats all!
n = 20

y = Vector{Float64}(undef,n-1)
for i =1:n-1
    y[i] = 0
end

function task_f()
    var = initialize()
    x = Vector{Float64}(undef,n)
    vn = @varname x[1]
    x[1] = update_var(var, vn, rand(Normal()))
    report_transition!(var,0.0,0.0)
    for i = 2:n
        # Sampling
        r =  rand(Normal())
        vn = @varname x[i]
        x[i] = update_var(var, vn, r)
        logγ = logpdf(Normal(),x[i]) #γ(x_t|x_t-1)
        logp = logpdf(Normal(),x[i])                # p(x_t|x_t-1)
        report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i], 1.0), y[i-1])
        var = report_observation!(var,logpy)
    end
end



task = create_task(task_f)

model = PFModel(task)

tonamedtuple(vi::UntypedVarInfo) = tonamedtuple(TypedVarInfo(vi))

alg = AdvancedPS.SMCAlgorithm()
uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
container = VarInfo()

chn =sample(model, alg, uf, container, 2000)




alg = AdvancedPS.PGAlgorithm(5)
uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
n = 5
T = Trace{typeof(container), PGTaskInfo{Float64}}

# if hasfield(typeof(container),:ref_traj) && spl.ref_traj !== nothing
#     particles = T[ Trace(container, model.task, SMCTaskInfo(), uf.copy) for _ =1:n-1]
#     pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n),0.0,0)
#
#     # Reset Task
#     spl.ref_traj.task = copy(model.task)
#
#     push!(pc, .ref_traj)
# else
particles = T[ Trace(container, model.task, PGTaskInfo(), uf.copy) for _ =1:n]
pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n),0.0,0)


AdvancedPS.sample!(pc, alg, uf, nothing)

ref_traj = pc[1]

particles = T[ Trace(container, model.task, PGTaskInfo(), uf.copy) for _ =1:n-1]
pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n),0.0,0)

# Reset Task
ref_traj = AdvancedPS.forkr(ref_traj, uf.copy)

push!(pc, ref_traj)
pc

AdvancedPS.sample!(pc, alg, uf, ref_traj)

container
vn = @varname vi
push!(container,vn,3.0,Normal())
container.metadata.flags



pc



indx = AdvancedPS.randcat(weights(pc))
particle = spl.ref_traj = pc[indx]
params = spl.uf.tonamedtuple(particle.vi)
return PFTransition(params, particle.taskinfo.logp, pc.logE, weights(pc)[indx])
