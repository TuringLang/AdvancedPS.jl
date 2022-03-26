@testset "container.jl" begin
    @testset "copy particle container" begin
        pc = AdvancedPS.ParticleContainer(AdvancedPS.Trace[])
        newpc = copy(pc)

        @test newpc.logWs == pc.logWs
        @test typeof(pc) === typeof(newpc)
    end

    @testset "particle container" begin
        # Create a resumable function that always returns the same log probability.
        function fpc(logp)
            f = let logp = logp
                rng -> begin
                    while true
                        produce(logp)
                    end
                end
            end
            return f
        end

        # Create particle container.
        logps = [0.0, -1.0, -2.0]
        particles = map(logps) do logp
            trng = AdvancedPS.TracedRNG()
            tmodel = AdvancedPS.GenericModel(fpc(logp), trng)
            AdvancedPS.Trace(tmodel, trng)
        end
        pc = AdvancedPS.ParticleContainer(particles)

        # Initial state.
        @test pc.logWs == zeros(3)
        @test AdvancedPS.getweights(pc) == fill(1 / 3, 3)
        @test all(AdvancedPS.getweight(pc, i) == 1 / 3 for i in 1:3)
        @test AdvancedPS.logZ(pc) ≈ log(3)
        @test AdvancedPS.effectiveSampleSize(pc) == 3

        # Reweight particles.
        AdvancedPS.reweight!(pc)
        @test pc.logWs == logps
        @test AdvancedPS.getweights(pc) ≈ exp.(logps) ./ sum(exp, logps)
        @test all(
            AdvancedPS.getweight(pc, i) ≈ exp(logps[i]) / sum(exp, logps) for i in 1:3
        )
        @test AdvancedPS.logZ(pc) ≈ log(sum(exp, logps))

        # Reweight particles.
        AdvancedPS.reweight!(pc)
        @test pc.logWs == 2 .* logps
        @test AdvancedPS.getweights(pc) == exp.(2 .* logps) ./ sum(exp, 2 .* logps)
        @test all(
            AdvancedPS.getweight(pc, i) ≈ exp(2 * logps[i]) / sum(exp, 2 .* logps) for
            i in 1:3
        )
        @test AdvancedPS.logZ(pc) ≈ log(sum(exp, 2 .* logps))

        # Resample and propagate particles with reference particle
        particles_ref = map(logps) do logp
            trng = AdvancedPS.TracedRNG()
            tmodel = AdvancedPS.GenericModel(fpc(logp), trng)
            AdvancedPS.Trace(tmodel, trng)
        end
        pc_ref = AdvancedPS.ParticleContainer(particles_ref)

        ref = particles_ref[end]
        AdvancedPS.advance!(ref) # Make sure ref has a valid history

        AdvancedPS.resample_propagate!(
            Random.GLOBAL_RNG, pc_ref, AdvancedPS.resample_systematic, ref
        )
        @test pc_ref.logWs == zeros(3)
        @test AdvancedPS.getweights(pc_ref) == fill(1 / 3, 3)
        @test all(AdvancedPS.getweight(pc_ref, i) == 1 / 3 for i in 1:3)
        @test AdvancedPS.logZ(pc_ref) ≈ log(3)
        @test AdvancedPS.effectiveSampleSize(pc_ref) == 3
        @test pc_ref.vals[end] === particles_ref[end]

        # Resample and propagate particles.
        AdvancedPS.resample_propagate!(Random.GLOBAL_RNG, pc)
        @test pc.logWs == zeros(3)
        @test AdvancedPS.getweights(pc) == fill(1 / 3, 3)
        @test all(AdvancedPS.getweight(pc, i) == 1 / 3 for i in 1:3)
        @test AdvancedPS.logZ(pc) ≈ log(3)
        @test AdvancedPS.effectiveSampleSize(pc) == 3

        # Reweight particles.
        AdvancedPS.reweight!(pc)
        @test pc.logWs ⊆ logps
        @test AdvancedPS.getweights(pc) == exp.(pc.logWs) ./ sum(exp, pc.logWs)
        @test all(
            AdvancedPS.getweight(pc, i) ≈ exp(pc.logWs[i]) / sum(exp, pc.logWs) for i in 1:3
        )
        @test AdvancedPS.logZ(pc) ≈ log(sum(exp, pc.logWs))

        # Increase unnormalized logarithmic weights.
        logws = copy(pc.logWs)
        AdvancedPS.increase_logweight!(pc, 2, 1.41)
        @test pc.logWs == logws + [0, 1.41, 0]

        # Reset unnormalized logarithmic weights.
        logws = pc.logWs
        AdvancedPS.reset_logweights!(pc)
        @test pc.logWs === logws
        @test all(iszero, pc.logWs)
    end

    @testset "trace" begin
        n = Ref(0)
        function f2(rng)
            t = TArray(Int, 1)
            t[1] = 0
            while true
                n[] += 1
                produce(t[1])
                n[] += 1
                t[1] = 1 + t[1]
            end
        end

        # Test task copy version of trace
        trng = AdvancedPS.TracedRNG()
        tmodel = AdvancedPS.GenericModel(f2, trng)
        tr = AdvancedPS.Trace(tmodel, trng)

        consume(tr.model.ctask)
        consume(tr.model.ctask)

        a = AdvancedPS.fork(tr)
        consume(a.model.ctask)
        consume(a.model.ctask)

        @test consume(tr.model.ctask) == 2
        @test consume(a.model.ctask) == 4
    end

    @testset "seed container" begin
        function dummy(rng) end

        seed = 1
        n = 3
        rng = Random.MersenneTwister(seed)

        particles = map(1:n) do _
            trng = AdvancedPS.TracedRNG()
            tmodel = AdvancedPS.GenericModel(dummy, trng)
            AdvancedPS.Trace(tmodel, trng)
        end
        pc = AdvancedPS.ParticleContainer(particles, AdvancedPS.TracedRNG())

        AdvancedPS.seed_from_rng!(pc, rng)
        old_seeds = vcat([part.rng.rng.key for part in pc.vals], [pc.rng.rng.key])

        Random.seed!(rng, seed)
        AdvancedPS.seed_from_rng!(pc, rng)
        new_seeds = vcat([part.rng.rng.key for part in pc.vals], [pc.rng.rng.key])

        # Check if we reset the seeds properly
        @test old_seeds ≈ new_seeds

        Random.seed!(rng, 2)
        AdvancedPS.seed_from_rng!(pc, rng, pc.vals[n])
        ref_seeds = vcat([part.rng.rng.key for part in pc.vals], [pc.rng.rng.key])

        # Dont reset reference particle
        @test ref_seeds[n] ≈ new_seeds[n]
    end
end
