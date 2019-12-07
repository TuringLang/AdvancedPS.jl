using Random
using Test
using Distributions
using Turing
using AdvancedPS

dir = splitdir(splitdir(pathof(AdvancedPS))[1])[1]

include(dir*"/Example/Using_Turing_VI/turing_interface.jl")
include(dir*"/Tests/test_utils/AllUtils.jl")

import Turing.Core: tonamedtuple
tonamedtuple(vi::UntypedVarInfo) = tonamedtuple(TypedVarInfo(vi))


@testset "apf.jl" begin
    @apf_testset "apf constructor" begin
        N = 200
        f = aps_gdemo_default
        task = create_task(f)
        model = PFModel(task)
        tcontainer = VarInfo()

        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)

        alg = AdvancedPS.PGAlgorithm(5)
        uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)

    end
    @numerical_testset "apf inference" begin
        N = 2000
        f = aps_gdemo_default
        task = create_task(f)
        model = PFModel(task)
        tcontainer = VarInfo()

        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_gdemo(caps1, atol = 0.1)

        alg = AdvancedPS.PGAlgorithm(5)
        uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_gdemo(caps1, atol = 0.1)
    end
end





@testset "test_against_turing.jl" begin

    @numerical_testset "apf inference" begin
        N = 5000


        ## Testset1
        y = [0 for i in 1:10]

        # We use the Turing model as baseline
        chn_base = sample(large_demo(y),PG(20),N)

        #############################################
        # Using a Proposal                          #
        #############################################
        f = large_demo_apf_proposal
        task = create_task(f,y)
        model = PFModel(task)
        tcontainer = VarInfo()

        ##SMC
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, 5*N)
        check_numerical(caps1,chn_base, 0.1, 0.2)

        # PG
        alg = AdvancedPS.PGAlgorithm(20)
        uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base,0.1, 0.2)



        #############################################
        # No Proposal                               #
        #############################################
        f = large_demo_apf
        task = create_task(f,y)
        model = PFModel(task)
        tcontainer = VarInfo()

        #SMC
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, 5*N)
        check_numerical(caps1,chn_base, 0.1, 0.2)

        # PG
        alg = AdvancedPS.PGAlgorithm(20)
        uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base,0.1, 0.2)



        ## Testset2
        y = [-0.1*i for i in 1:10]

        # We use the Turing model as baseline
        chn_base = sample(large_demo(y),PG(20),N)

        #############################################
        # Using a Proposal                          #
        #############################################
        f = large_demo_apf_proposal
        task = create_task(f,y)
        model = PFModel(task)
        tcontainer = VarInfo()

        #SMC
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, 5*N)
        check_numerical(caps1,chn_base,0.1, 0.2)

        # PG
        alg = AdvancedPS.PGAlgorithm(20)
        uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base, 0.1, 0.2)


        #############################################
        # No Proposal                               #
        #############################################
        f = large_demo_apf
        task = create_task(f,y)
        model = PFModel(task)
        tcontainer = VarInfo()

        #SMC
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, 5*N)
        check_numerical(caps1,chn_base,0.21, 0.2)

        # PG
        alg = AdvancedPS.PGAlgorithm(20)
        uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base, 0.1, 0.2)
    end
end
