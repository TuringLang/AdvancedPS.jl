using AdvancedSMC
using Test

@testset "particlecontainer.jl" begin
    @testset "copy particle container" begin
        pc = ParticleContainer(Trace[])
        newpc = copy(pc)

        @test newpc.logWs == pc.logWs
        @test typeof(pc) === typeof(newpc)
    end

    @testset "particle container" begin
        # Create a resumable function that always yields `logp`.
        function fpc(logp)
            f = let logp = logp
                () -> begin
                    while true
                        produce(logp)
                    end
                end
            end
            return f
        end

        # Dummy sampler that is not actually used.
        sampler = Sampler(PG(5), empty_model())

        # Create particle container.
        logps = [0.0, -1.0, -2.0]
        particles = [Trace(fpc(logp), empty_model(), sampler, VarInfo()) for logp in logps]
        pc = ParticleContainer(particles)

        # Initial state.
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs == zeros(3)
        @test getweights(pc) == fill(1/3, 3)
        @test all(getweight(pc, i) == 1/3 for i in 1:3)
        @test logZ(pc) ≈ log(3)
        @test effectiveSampleSize(pc) == 3

        # Reweight particles.
        reweight!(pc)
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs == logps
        @test getweights(pc) ≈ exp.(logps) ./ sum(exp, logps)
        @test all(getweight(pc, i) ≈ exp(logps[i]) / sum(exp, logps) for i in 1:3)
        @test logZ(pc) == log(sum(exp, logps))

        # Reweight particles.
        reweight!(pc)
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs == 2 .* logps
        @test getweights(pc) == exp.(2 .* logps) ./ sum(exp, 2 .* logps)
        @test all(getweight(pc, i) ≈ exp(2 * logps[i]) / sum(exp, 2 .* logps) for i in 1:3)
        @test logZ(pc) == log(sum(exp, 2 .* logps))

        # Resample and propagate particles.
        resample_propagate!(pc)
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs == zeros(3)
        @test getweights(pc) == fill(1/3, 3)
        @test all(getweight(pc, i) == 1/3 for i in 1:3)
        @test logZ(pc) ≈ log(3)
        @test effectiveSampleSize(pc) == 3

        # Reweight particles.
        reweight!(pc)
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs ⊆ logps
        @test getweights(pc) == exp.(pc.logWs) ./ sum(exp, pc.logWs)
        @test all(getweight(pc, i) ≈ exp(pc.logWs[i]) / sum(exp, pc.logWs) for i in 1:3)
        @test logZ(pc) == log(sum(exp, pc.logWs))

        # Increase unnormalized logarithmic weights.
        logws = copy(pc.logWs)
        increase_logweight!(pc, 2, 1.41)
        @test pc.logWs == logws + [0, 1.41, 0]

        # Reset unnormalized logarithmic weights.
        logws = pc.logWs
        reset_logweights!(pc)
        @test pc.logWs === logws
        @test all(iszero, pc.logWs)
    end
end
