
## It is not yet a package...

using AdvancedPS
using Libtask
const APS = AdvancedPS
using Distributions
using Plots

n = 20
include("AdvancedPS.jl/Example/test_interface.jl")


## Our states
vi = Container(zeros(n),[false for i =1:n] ,zeros(n), 0)


## Our observations
y = Vector{Float64}(undef,n)
for i = 1:n-1
    y[i] = -i
end

# Define a short model.
# The syntax is rather simple. Observations need to be reported with report_observation.
# Transitions must be reported using report_transition.
# The trace contains the variables which we want to infer using particle gibbs.
# Thats all!
function task_f()

    var = current_trace()
    set_x!(var,1,rand(Normal())) # We sample
    report_transition!(var,0.0,0.0)

    for i = 2:n
        # Sampling
        r =  rand(Normal(get_x(var,i-1)-1,0.8))
        set_x!(var,i,r) # We sample from proposal
        logγ = logpdf(Normal(get_x(var,i-1)-1,0.8),get_x(var,i)) #γ(x_t|x_t-1)
        logp = logpdf(Normal(),get_x(var,i))                # p(x_t|x_t-1)


        report_transition!(var,logp,logγ)

        #Proposal and Resampling
        logpy = logpdf(Normal(get_x(var,i),0.4),y[i-1])

        var = report_observation!(var,logpy)

    end


end

m = 10
task = create_task(task_f)

particlevec = [Trace(vi, task, PGTaskInfo(0.0,0.0),deepcopy) for i =1:m]
particlevec
ws = [0 for i =1:m]

particles = APS.ParticleContainer{typeof(particlevec[1])}(particlevec,deepcopy(ws),deepcopy(ws),0,0)

Algo = APS.SMCAlgorithm(APS.resample_systematic,0.0,APS.SMCUtilityFunctions(copy_container,set_retained_vns_del_by_spl!))
## Do one SMC step.
vi
APS.sample!(particles,Algo)



particles

means = [mean([sum(particles.vals[i].vi.x[2:20]-y[1:19]) for i =1:m])]






particles




particlevec = [Trace(vi, task, PGTaskInfo(0.0,0.0),deepcopy) for i =1:m-1]
push!(particlevec,Trace(particles.vals[1].vi,task,PGTaskInfo(0.0,0.0),deepcopy))
ws = [0 for i =1:m]
particles = APS.ParticleContainer{typeof(particlevec[1])}(particlevec,deepcopy(ws),deepcopy(ws),0,0)

Algo = APS.PGAlgorithm(APS.resample_systematic,0.0,APS.PGUtilityFunctions(copy_container,set_retained_vns_del_by_spl!))
## Do one SMC step.
APS.sample!(particles,Algo)

means = [mean([sum(particles.vals[i].vi.x[2:20]-y[1:19]) for i =1:m])]
