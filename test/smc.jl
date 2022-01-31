@testset "smc.jl" begin
    @testset "SMC constructor" begin
        sampler = AdvancedPS.SMC(10)
        @test sampler.nparticles == 10
        @test sampler.resampler == AdvancedPS.ResampleWithESSThreshold()

        sampler = AdvancedPS.SMC(15, 0.6)
        @test sampler.nparticles == 15
        @test sampler.resampler ===
              AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_systematic, 0.6)

        sampler = AdvancedPS.SMC(20, AdvancedPS.resample_multinomial, 0.6)
        @test sampler.nparticles == 20
        @test sampler.resampler ===
              AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_multinomial, 0.6)

        sampler = AdvancedPS.SMC(25, AdvancedPS.resample_systematic)
        @test sampler.nparticles == 25
        @test sampler.resampler === AdvancedPS.resample_systematic
    end

    # Smoke tests
    @testset "models" begin
        mutable struct NormalModel <: AbstractMCMC.AbstractModel
            a::Float64
            b::Float64

            NormalModel() = new()
        end

        function (m::NormalModel)(rng::Random.AbstractRNG)
            # First latent variable.
            m.a = a = rand(rng, Normal(4, 5))

            # First observation.
            AdvancedPS.observe(Normal(a, 2), 3)

            # Second latent variable.
            m.b = b = rand(rng, Normal(a, 1))

            # Second observation.
            return AdvancedPS.observe(Normal(b, 2), 1.5)
        end

        sample(NormalModel(), AdvancedPS.SMC(100))

        # failing test
        mutable struct FailSMCModel <: AbstractMCMC.AbstractModel
            a::Float64
            b::Float64

            FailSMCModel() = new()
        end

        function (m::FailSMCModel)(rng::Random.AbstractRNG)
            m.a = a = rand(rng, Normal(4, 5))
            m.b = b = rand(rng, Normal(a, 1))
            if a >= 4
            AdvancedPS.observe(Normal(b, 2), 1.5)
            end
        end

        @test_throws ErrorException sample(FailSMCModel(), AdvancedPS.SMC(100))
    end

    @testset "logevidence" begin
        Random.seed!(100)

        mutable struct TestModel <: AbstractMCMC.AbstractModel
            a::Float64
            x::Bool
            b::Float64
            c::Float64

            TestModel() = new()
        end

        function (m::TestModel)(rng::Random.AbstractRNG)
            # First hidden variables.
            m.a = rand(rng, Normal(0, 1))
            m.x = x = rand(rng, Bernoulli(1))
            m.b = rand(rng, Gamma(2, 3))

            # First observation.
            AdvancedPS.observe(Bernoulli(x / 2), 1)

            # Second hidden variable.
            m.c = rand(rng, Beta())

            # Second observation.
            return AdvancedPS.observe(Bernoulli(x / 2), 0)
        end

        chains_smc = sample(TestModel(), AdvancedPS.SMC(100))

        @test all(isone(p.f.x) for p in chains_smc.trajectories)
        @test chains_smc.logevidence ≈ -2 * log(2)
    end

    @testset "PG constructor" begin
        sampler = AdvancedPS.PG(10)
        @test sampler.nparticles == 10
        @test sampler.resampler == AdvancedPS.ResampleWithESSThreshold()

        sampler = AdvancedPS.PG(60, 0.6)
        @test sampler.nparticles == 60
        @test sampler.resampler ===
              AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_systematic, 0.6)

        sampler = AdvancedPS.PG(80, AdvancedPS.resample_multinomial, 0.6)
        @test sampler.nparticles == 80
        @test sampler.resampler ===
              AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_multinomial, 0.6)

        sampler = AdvancedPS.PG(100, AdvancedPS.resample_systematic)
        @test sampler.nparticles == 100
        @test sampler.resampler === AdvancedPS.resample_systematic
    end

    @testset "logevidence" begin
        Random.seed!(100)

        mutable struct TestModel <: AbstractMCMC.AbstractModel
            a::Float64
            x::Bool
            b::Float64
            c::Float64

            TestModel() = new()
        end

        function (m::TestModel)(rng::Random.AbstractRNG)
            # First hidden variables.
            m.a = rand(rng, Normal(0, 1))
            m.x = x = rand(rng, Bernoulli(1))
            m.b = rand(rng, Gamma(2, 3))

            # First observation.
            AdvancedPS.observe(Bernoulli(x / 2), 1)

            # Second hidden variable.
            m.c = rand(rng, Beta())

            # Second observation.
            return AdvancedPS.observe(Bernoulli(x / 2), 0)
        end

        chains_pg = sample(TestModel(), AdvancedPS.PG(10), 100)

        @test all(isone(p.trajectory.f.x) for p in chains_pg)
        @test mean(x.logevidence for x in chains_pg) ≈ -2 * log(2) atol = 0.01
    end

    @testset "Replay reference" begin
        mutable struct Model <: AbstractMCMC.AbstractModel
            a::Float64
            b::Float64

            Model() = new()
        end

        function (m::Model)(rng)
            m.a = rand(rng, Normal())
            AdvancedPS.observe(Normal(), m.a)

            m.b = rand(rng, Normal())
            return AdvancedPS.observe(Normal(), m.b)
        end

        pg = AdvancedPS.PG(1)
        first, second = sample(Model(), pg, 2)

        first_model = first.trajectory.f
        second_model = second.trajectory.f

        # Single Particle - must be replaying
        @test first_model.a ≈ second_model.a
        @test first_model.b ≈ second_model.b
        @test first.logevidence ≈ second.logevidence
    end
end

# @testset "pmmh.jl" begin
#     @turing_testset "pmmh constructor" begin
#         N = 2000
#         s1 = PMMH(N, SMC(10, :s), MH(1,(:m, s -> Normal(s, sqrt(1)))))
#         s2 = PMMH(N, SMC(10, :s), MH(1, :m))
#         s3 = PIMH(N, SMC())
#
#         c1 = sample(gdemo_default, s1)
#         c2 = sample(gdemo_default, s2)
#         c3 = sample(gdemo_default, s3)
#     end
#     @numerical_testset "pmmh inference" begin
#         alg = PMMH(2000, SMC(20, :m), MH(1, (:s, GKernel(1))))
#         chain = sample(gdemo_default, alg)
#         check_gdemo(chain, atol = 0.1)
#
#         # PMMH with prior as proposal
#         alg = PMMH(2000, SMC(20, :m), MH(1, :s))
#         chain = sample(gdemo_default, alg)
#         check_gdemo(chain, atol = 0.1)
#
#         # PIMH
#         alg = PIMH(2000, SMC())
#         chain = sample(gdemo_default, alg)
#         check_gdemo(chain)
#
#         # MoGtest
#         pmmh = PMMH(2000,
#             SMC(10, :z1, :z2, :z3, :z4),
#             MH(1, :mu1, :mu2))
#         chain = sample(MoGtest_default, pmmh)
#
#         check_MoGtest_default(chain, atol = 0.1)
#     end
# end

# @testset "ipmcmc.jl" begin
#     @turing_testset "ipmcmc constructor" begin
#         Random.seed!(125)
#
#         N = 50
#         s1 = IPMCMC(10, N, 4, 2)
#         s2 = IPMCMC(10, N, 4)
#
#         c1 = sample(gdemo_default, s1)
#         c2 = sample(gdemo_default, s2)
#     end
#     @numerical_testset "ipmcmc inference" begin
#         alg = IPMCMC(30, 500, 4)
#         chain = sample(gdemo_default, alg)
#         check_gdemo(chain)
#
#         alg2 = IPMCMC(15, 100, 10)
#         chain2 = sample(MoGtest_default, alg2)
#         check_MoGtest_default(chain2, atol = 0.2)
#     end
# end
