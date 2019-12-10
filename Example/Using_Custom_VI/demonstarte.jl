
## It is not yet a package...


using Distributions
using AdvancedPS
using BenchmarkTools

dir = splitdir(splitdir(pathof(AdvancedPS))[1])[1]
push!(LOAD_PATH,dir*"/Example/Using_Custom_VI/" )
using AdvancedPS_SSM_Container
const APSCont = AdvancedPS_SSM_Container


# Define a short model.
# The syntax is rather simple. Observations need to be reported with report_observation.
# Transitions must be reported using report_transition.
# The trace contains the variables which we want to infer using particle gibbs.
# Thats all!
n = 200


y = Array{Float64,2}(undef,n-1,1)
for i =1:n-1
    y[i,1] = 0
end



function task_f(y)
    var = initialize()
    x = zeros(n,1)
    r = rand(MultivariateNormal(1,1.0))
    x[1,:] = update_var!(var, 1,  r)
    report_transition!(var,0.0,0.0)
    for i = 2:n
        # Sampling
        r = rand(MultivariateNormal(1,1.0))
        x[i,:] = update_var!(var, i, r)
        logγ = logpdf(MultivariateNormal(1,1.0),x[i,:]) #γ(x_t|x_t-1)
        logp = logpdf(MultivariateNormal(1,1.0),x[i,:])                # p(x_t|x_t-1)
        report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(MultivariateNormal(x[i,:],1.0), y[i-1,:])
        var = report_observation!(var,logpy)
    end
end


tcontainer =  Container(zeros(n,1),Vector{Bool}(undef,n),Vector{Int}(zeros(n)),0)
model = PFModel(task_f, (y=y,))



alg = AdvancedPS.SMCAlgorithm()
uf = AdvancedPS.SMCUtilityFunctions(APSCont.set_retained_vns_del_by_spl!, tonamedtuple)
@btime sample(model, alg, uf, tcontainer, 10)


alg = AdvancedPS.PGAlgorithm(AdvancedPS.resample_systematic, 1.0, 10)
uf = AdvancedPS.PGUtilityFunctions( APSCont.set_retained_vns_del_by_spl!, APSCont.tonamedtuple)
@btime chn2 =sample(model, alg, uf, tcontainer, 5)
