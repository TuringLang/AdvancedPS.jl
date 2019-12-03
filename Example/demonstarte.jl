
## It is not yet a package...

using AdvancedPS
using Libtask
const APS = AdvancedPS
using Distributions
n = 20
include("AdvancedPS/Tests/testInterface.jl")



## Our states
vi = Container(zeros(n),0)
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
    var = initialize()
    set_x(var,1,rand(Normal())) # We sample
    report_transition(var,0.0,0.0)
    for i = 2:n
        # Sampling
        set_x(var,i, rand(Normal(get_x(var,i-1)-1,0.8))) # We sample from proposal
        logγ = logpdf(Normal(get_x(var,i-1)-1,0.8),get_x(var,i)) #γ(x_t|x_t-1)
        logp = logpdf(Normal(),get_x(var,i))                # p(x_t|x_t-1)


        report_transition(var,logp,logγ)

        #Proposal and Resampling
        logpy = logpdf(Normal(get_x(var,i),0.4),y[i-1])

        var = report_observation(var,logpy)

    end

end
task = create_task(task_f)
m = 10

particlevec = [Trace(deepcopy(vi), copy(task), PGTaskInfo(0.0,0.0)) for i =1:m]
ws = [0 for i =1:m]
particles = APS.ParticleContainer{typeof(particlevec[1])}(particlevec,deepcopy(ws),deepcopy(ws),0,0)

Algo = APS.SMCAlgorithm(APS.resample_systematic,0.0,APS.SMCUtilityFunctions(deepcopy,(x)-> x))
## Do one SMC step.
APS.sample!(particles,Algo)

particles2 = APS.ParticleContainer{typeof(vi),APS.PGTaskInfo }()


m = 10
task = create_task(task_f)


APS.push!(particles2, 9, vi, task, PGTaskInfo(0.0,0.0))
APS.push!(particles2, 1, particles[1].vi, task,  PGTaskInfo(0.0,0.0))

## Do one SMC step.
APS.samplePG!(particles2,resample_systematic,particles2[m])

particles2







a
function task_f_prod()
    var = init()

    set_x(var,1,rand(Normal())) # We sample
    arr = cu(rand(1000,1000))
    arr2 = cu(rand(1000,1))
    arr3 = arr*arr2
    for i = 2:n
        # Sampling
        set_x(var,i, rand(Normal(get_x(var,i-1)-1,0.8))) # We sample from proposal
        logγ = logpdf(Normal(get_x(var,i-1)-1,0.8),get_x(var,i)) #γ(x_t|x_t-1)
        logp = logpdf(Normal(),get_x(var,i))                # p(x_t|x_t-1)


        report_transition(var,logp,logγ)

        #Proposal and Resampling
        logpy = logpdf(Normal(get_x(var,i),0.4),y[i-1])

        var = report_observation(var,logpy)
        if var.taskinfo.hold
            produce(0.0)

    end
end
