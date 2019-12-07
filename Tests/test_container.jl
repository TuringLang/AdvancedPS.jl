
@testset "Particle.jl" begin
    @apf_testset "copy particle container" begin
        pc = ParticleContainer(Trace[], Float64[], 0.0, 0)
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, () -> (), () -> (), () ->())
        newpc = copy(pc, uf)
        @test newpc.logE        == pc.logE
        @test newpc.logWs       == pc.logWs
        @test newpc.n_consume   == pc.n_consume
        @test typeof(pc) === typeof(newpc)
    end

    @apf_testset "particle container" begin
        n = Ref(0)
        alg = AdvancedPS.PGAlgorithm(5)
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)

        dist = Normal(0, 1)

        function fpc()
            t = TArray(Float64, 1);
            var = initialize()
            t[1] = 0;
            while true
                vn = @varname x[n]
                r = rand(dist)
                update_var(var, vn, r)
                n[] += 1
                produce(0)
                var = current_trace()
                r = rand(dist)
                update_var(var, vn, r)
                t[1] = 1 + t[1]
            end
        end

        task = create_task(fpc)
        model = PFModel(task)
        particles = [Trace(Turing.VarInfo(), fpc, PGTaskInfo(0.0, 0.0), deepcopy) for _ in 1:3]
        pc = ParticleContainer(particles)

        @test weights(pc) == [1/3, 1/3, 1/3]
        @test logZ(pc) ≈ log(3)
        @test pc.logE ≈ log(1)
        @test consume(pc) == log(1)

        Ws = weights(pc)
        indx = alg.resampler(Ws, length(pc))
        resample!(pc, uf, indx, nothing)
        @test weights(pc) == [1/3, 1/3, 1/3]
        @test logZ(pc) ≈ log(3)
        @test pc.logE ≈ log(1)
        @test AdvancedPS.effectiveSampleSize(pc) == 3
        @test consume(pc) ≈ log(1)
        Ws = weights(pc)
        indx = alg.resampler(Ws, length(pc))
        resample!(pc, uf, indx, nothing)
        @test consume(pc) ≈ log(1)
    end


    @apf_testset "trace" begin
        n = Ref(0)
        alg = AdvancedPS.PGAlgorithm(5)
        uf = AdvancedPS.SMCUtilityFunctions(deepcopy, set_retained_vns_del_by_spl!, empty!, tonamedtuple)
        dist = Normal(0, 1)
        function f2()
            t = TArray(Float64, 1);
            var = initialize()
            t[1] = 0;
            while true
                vn = @varname x[n]
                r = rand(dist)
                update_var(var, vn, r)
                produce(t[1]);
                var = current_trace()
                vn = @varname x[n]
                r = rand(dist)
                update_var(var, vn, r)
                t[1] = 1 + t[1]
            end
        end
        task = create_task(f2)
        model = PFModel(task)
        tr = Trace(Turing.VarInfo(), f2, PGTaskInfo(0.0, 0.0), deepcopy)
        consume(tr); consume(tr)
        a = AdvancedPS.fork(tr, deepcopy);
        consume(a); consume(a)
        @test consume(tr) == 2
        @test consume(a) == 4

    end
end
