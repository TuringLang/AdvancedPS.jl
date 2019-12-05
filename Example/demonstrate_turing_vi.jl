
## It is not yet a package...

using AdvancedPS
using Libtask
const APS = AdvancedPS
using Distributions
using Plots
using Turing.Core.RandomVariables
import Turing.Core:  @varname
include("AdvancedPS.jl/Example/turing_interface.jl")
import Turing: SampleFromPrior
using StatsFuns: softmax!
using AbstractMCMC
# Define a short model.
# The syntax is rather simple. Observations need to be reported with report_observation.
# Transitions must be reported using report_transition.
# The trace contains the variables which we want to infer using particle gibbs.
# Thats all!
n = 20

y = Vector{Float64}(undef,n-1)
for i =1:n-1
    y[i] = sin(0.1*i)
end

function task_f()
    var = initialize()
    x = Vector{Float64}(undef,n)
    vn = @varname x[1]
    x[1] = update_var(var, vn, rand(Normal()))
    report_transition!(var,0.0,0.0)
    for i = 2:n
        # Sampling
        r =  rand(Normal(x[i-1],0.2))
        vn = @varname x[i]
        x[i] = update_var(var, vn, r)
        logγ = logpdf(Normal(x[i-1],0.2),x[i]) #γ(x_t|x_t-1)
        logp = logpdf(Normal(),x[i])                # p(x_t|x_t-1)
        report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i],0.2),y[i-1])
        var = report_observation!(var,logpy)
    end
end


task = create_task(task_f)




### This is for initilaization!!
m = 1
vi_c = VarInfo()
task
particlevec = [APS.Trace(vi_c, task, PGTaskInfo(0.0,0.0),deepcopy) for i =1:m]
particlevec
ws = [0 for i =1:m]

particles = APS.ParticleContainer{typeof(particlevec[1])}(particlevec,deepcopy(ws),zeros(m),0,0)


Algo = APS.SMCAlgorithm(APS.resample_systematic,1.0,APS.SMCUtilityFunctions(deepcopy,set_retained_vns_del_by_spl!))
## Do one SMC step.

@elapsed APS.sample!(particles,Algo)

vi_c = empty!(VarInfo{<:NamedTuple}(particles[1].vi))

const vi_ct = vi_c


m = 200
task = create_task(task_f)
task
particlevec = [APS.Trace(vi_c, task, PGTaskInfo(0.0,0.0),deepcopy) for i =1:m]
particlevec
ws = [0 for i =1:m]

particles = APS.ParticleContainer{typeof(particlevec[1])}(particlevec,deepcopy(ws),zeros(m),0,0)


Algo = APS.SMCAlgorithm(APS.resample_systematic,0.5,APS.SMCUtilityFunctions(deepcopy,set_retained_vns_del_by_spl!))
## Do one SMC step.


@elapsed APS.sample!(particles,Algo)



particles

#means = [mean([sum(particles.vals[i].vi.vals[2:20]-y[1:19])^2 for i =1:m])]


set_retained_vns_del_by_spl!(vi_c, SampleFromPrior())
for l = 2:10
    global particles
    indx = AdvancedPS.randcat(softmax!(copy(particles.logWs)))

    println(indx)
    particlevec = [Trace(vi_c, task, PGTaskInfo(0.0,0.0),deepcopy) for i =1:m]

    #push!(particlevec,Trace(particles.vals[indx].vi, task, PGTaskInfo(0.0,0.0), deepcopy))
    ws = [0 for i =1:m]
    particles = APS.ParticleContainer{typeof(particlevec[1])}(particlevec,deepcopy(ws),deepcopy(ws),0,0)
    Algo = APS.PGAlgorithm(APS.resample_systematic,1.0,APS.SMCUtilityFunctions(deepcopy,set_retained_vns_del_by_spl!))
    ## Do one SMC step.
    APS.sample!(particles,Algo,particles[indx])

    push!(means, mean([sum(particles.vals[i].vi.vals[2:20]-y[1:19])^2 for i =1:m]))
end

particles

plot(means)

means
