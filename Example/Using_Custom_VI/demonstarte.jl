
## It is not yet a package...
using Distributions
using AdvancedPS
using Libtask
using BenchmarkTools
using NamedTupleTools
include("AdvancedPS/Example/Using_Custom_VI/test_interface.jl")
# Define a short model.
# The syntax is rather simple. Observations need to be reported with report_observation.
# Transitions must be reported using report_transition.
# The trace contains the variables which we want to infer using particle gibbs.
# Thats all!
n = 3000

y = Vector{Float64}(undef,n-1)
for i =1:n-1
    y[i] = 0
end

function task_f(y)
    var = initialize()
    x = zeros(n,1)
    r = vectorize(Normal(), rand(Normal()))
    x[1,:] = update_var(var, 1,  r)
    report_transition!(var,0.0,0.0)
    for i = 2:n
        # Sampling
        r =  vectorize(Normal(),rand(Normal()))
        x[i,:] = update_var(var, i, r)
        logγ = logpdf(Normal(),x[i,1]) #γ(x_t|x_t-1)
        logp = logpdf(Normal(),x[i,1])                # p(x_t|x_t-1)
        report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i,1], 1.0), y[i-1])
        var = report_observation!(var,logpy)
    end
end





task = create_task(task_f, y)
model = PFModel(task)
tcontainer =  Container(zeros(n,1),Vector{Bool}(undef,n),zeros(n),0)



alg = AdvancedPS.SMCAlgorithm()
uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
@btime sample(model, alg, uf, tcontainer, 100)


alg = AdvancedPS.PGAlgorithm(AdvancedPS.resample_systematic, 1.0, 10)
uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
@btime chn2 =sample(model, alg, uf, tcontainer, 5)
