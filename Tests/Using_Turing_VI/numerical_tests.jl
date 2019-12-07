
@testset "apf.jl" begin
    @apf_testset "apf constructor" begin
        N = 200
        f = aps_gdemo_default
        task = create_task(f)
        model = PFModel(task)
        tcontainer = VarInfo()

        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        sample(model, alg, uf, tcontainer, N)

        alg = AdvancedPS.PGAlgorithm(5)
        uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        sample(model, alg, uf, tcontainer, N)

    end
    @numerical_testset "apf inference" begin
        N = 2000
        f = aps_gdemo_default
        task = create_task(f)
        model = PFModel(task)
        tcontainer = VarInfo()

        alg = AdvancedPS.PGAlgorithm(5)
        uf = AdvancedPS.PGUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_gdemo(caps1, atol = 0.1)

        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_gdemo(caps1, atol = 0.1)
    end
end
