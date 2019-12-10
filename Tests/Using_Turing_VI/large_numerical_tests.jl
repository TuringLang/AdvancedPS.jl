
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
        model = PFModel(f, (y=y, ))
        tcontainer = VarInfo()

        ##SMC
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions( set_retained_vns_del_by_spl!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, 5*N)
        check_numerical(caps1,chn_base, 0.1, 0.2)

        # PG
        alg = AdvancedPS.PGAlgorithm(20)
        uf = AdvancedPS.PGUtilityFunctions( set_retained_vns_del_by_spl!,  tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base,0.1, 0.2)



        #############################################
        # No Proposal                               #
        #############################################
        f = large_demo_apf
        model = PFModel(f, (y=y, ))
        tcontainer = VarInfo()

        #SMC
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions( set_retained_vns_del_by_spl!,  tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, 5*N)
        check_numerical(caps1,chn_base, 0.1, 0.2)

        # PG
        alg = AdvancedPS.PGAlgorithm(20)
        uf = AdvancedPS.PGUtilityFunctions( set_retained_vns_del_by_spl!,  tonamedtuple)
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
        model = PFModel(f, (y=y, ))
        tcontainer = VarInfo()

        #SMC
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions( set_retained_vns_del_by_spl!,  tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, 5*N)
        check_numerical(caps1,chn_base,0.1, 0.2)

        # PG
        alg = AdvancedPS.PGAlgorithm(20)
        uf = AdvancedPS.PGUtilityFunctions(set_retained_vns_del_by_spl!,  tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base, 0.1, 0.2)


        #############################################
        # No Proposal                               #
        #############################################
        f = large_demo_apf
        model = PFModel(f, (y=y, ))
        tcontainer = VarInfo()

        #SMC
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions( set_retained_vns_del_by_spl!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, 5*N)
        check_numerical(caps1,chn_base,0.21, 0.2)

        # PG
        alg = AdvancedPS.PGAlgorithm(20)
        uf = AdvancedPS.PGUtilityFunctions( set_retained_vns_del_by_spl!,  tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base, 0.1, 0.2)
    end
end
