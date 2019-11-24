## It is not yet a package...
using AdvancedPS
using Libtask
const APS = AdvancedPS
using Distributions

n = 20



## Our states
vi = Container(zeros(n),0)
## Our observations
y = Vector{Float64}(undef,n)
for i = 1:n-1
    y[i] = -i
end
## The particle container
particles = APS.ParticleContainer{typeof(vi),APS.PGTaskInfo }()

# Define a short model.
# The syntax is rather simple. Observations need to be reported with report_observation.
# Transitions must be reported using report_transition.
# The trace contains the variables which we want to infer using particle gibbs.
# Thats all!
function task_f()
    var = init()

    set_x(var,1,rand(Normal())) # We sample
    logpseq += logpdf(Normal(),get_x(var,1))

    for i = 2:n
        ## Sampling
        set_x(var,i, rand(Normal(get_x(var,i-1)-1,0.8))) # We sample from proposal
        logγ = logpdf(Normal(get_x(var,i-1)-1,0.8)),get_x(var,i)) #γ(x_t|x_t-1)
        logp = logpdf(Normal(),get_x(var,i))                # p(x_t|x_t-1)
        report_transition!(var,logp,logγ)

        #Proposal and Resampling
        logpy = logpdf(Normal(get_x(var,i),0.4),y[i-1])
        report_observation!(var,logpy)
    end
end


#


m = 10
task = create_task(task_f)
APS.extend!(particles, 10, vi, task, PGTaskInfo(0.0,0.0))

## Do one SMC step.
APS.sampleSMC!(particles,0.0)
