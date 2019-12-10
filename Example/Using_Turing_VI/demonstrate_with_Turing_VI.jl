
## It is not yet a package...
using Distributions
using AdvancedPS
using BenchmarkTools
using Libtask
const APS = AdvancedPS
using Turing
using Turing.Core: @varname


dir = splitdir(splitdir(pathof(AdvancedPS))[1])[1]
push!(LOAD_PATH,dir*"/Example/Using_Turing_VI/" )
using AdvancedPS_Turing_Container
const APSTCont = AdvancedPS_Turing_Container

# Define a short model.
# The syntax is rather simple. Observations need to be reported with report_observation.
# Transitions must be reported using report_transition.
# The trace contains the variables which we want to infer using particle gibbs.
# Thats all!
n = 50

y = Vector{Float64}(undef,n-1)
for i =1:n-1
    y[i] = 0
end

function task_f(y)
    var = initialize()
    x = TArray{Float64}(undef,n)
    vn = @varname x[1]
    x[1] = update_var!(var, vn, rand(Normal()))
    report_transition!(var,0.0,0.0)
    for i = 2:n
        # Sampling
        r =  rand(Normal())
        vn = @varname x[i]
        x[i] = update_var!(var, vn, r)
        logγ = logpdf(Normal(),x[i]) #γ(x_t|x_t-1)
        logp = logγ             # p(x_t|x_t-1)
        report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i], 1.0), y[i-1])
        var = report_observation!(var,logpy)
    end
end




model = PFModel(task_f, (y=y,))


#################################################################
# Get type stability!! - This can also be omitted               #
#################################################################
alg = AdvancedPS.SMCAlgorithm()
uf = AdvancedPS.SMCUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!, APSTCont.tonamedtuple)

untypedcontainer = VarInfo()
T = Trace{typeof(untypedcontainer),SMCTaskInfo{Float64}}
particles = T[ APS.get_new_trace(untypedcontainer, model.task, SMCTaskInfo()) for _ =1:1]
pc = ParticleContainer{typeof(particles[1])}(particles,zeros(1),0.0,0)
AdvancedPS.sample!(pc, alg, uf, nothing)
pc
container = empty!(TypedVarInfo(pc[1].vi))
tcontainer = container

#################################################################
# Now lets start                                      #
#################################################################



alg = AdvancedPS.SMCAlgorithm()
uf = AdvancedPS.SMCUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!, APSTCont.tonamedtuple)
@btime sample(model, alg, uf, tcontainer, 10)




alg = AdvancedPS.PGAlgorithm(AdvancedPS.resample_systematic, 1.0, 10)
uf = AdvancedPS.PGUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!, APSTCont.tonamedtuple)
@elapsed chn2 =sample(model, alg, uf, tcontainer, 5)
