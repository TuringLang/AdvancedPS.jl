## It is not yet a package...
using AdvancedPS
using Libtask
APS = AdvancedPS
using Distributions
n = 20





## Our states
vi =    APS.Container(zeros(n),0)
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
    var = APS.init()
    APS.set_x(var,1,rand(Normal())) # We sample
    APS.report_transition(var,0.0,0.0)
    for i = 2:n
        # Sampling
        APS.set_x(var,i, rand(Normal(APS.get_x(var,i-1)-1,0.8))) # We sample from proposal
        logγ = logpdf(Normal(APS.get_x(var,i-1)-1,0.8),APS.get_x(var,i)) #γ(x_t|x_t-1)
        logp = logpdf(Normal(),APS.get_x(var,i))                # p(x_t|x_t-1)


        APS.report_transition(var,logp,logγ)

        #Proposal and Resampling
        logpy = logpdf(Normal(APS.get_x(var,i),0.4),y[i-1])

        var = APS.report_observation(var,logpy)
        if var.taskinfo.hold
            produce(0.0)
        end

    end

end



particles = APS.ParticleContainer{typeof(vi),APS.PGASTaskInfo }()


m = 10
task = APS.create_task(task_f)


APS.extend!(particles, 10, vi, task, APS.PGASTaskInfo(0.0,0.0))
## Do one SMC step.
APS.samplePGAS!(particles)

particles2 = APS.ParticleContainer{typeof(vi),APS.PGASTaskInfo }()


m = 10
task = APS.create_task(task_f)


APS.extend!(particles2, 9, vi, task, APS.PGASTaskInfo(0.0,0.0,true))
APS.extend!(particles2, 1, particles[1].vi, task,  APS.PGASTaskInfo(0.0,0.0,true))

particles2.manipulators["merge_traj"] = (x,y,i=0) -> y  # Obviously this function is wrong!
## Do one SMC step.
APS.samplePGAS!(particles2,APS.resample_systematic,particles2[m])

particles2
