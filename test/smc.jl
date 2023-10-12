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
        mutable struct NormalModel <: AdvancedPS.AbstractGenericModel
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
        mutable struct FailSMCModel <: AdvancedPS.AbstractGenericModel
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

        mutable struct TestModel <: AdvancedPS.AbstractGenericModel
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

        @test all(isone(particle.x) for particle in chains_smc.trajectories)
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

        mutable struct TestModel <: AdvancedPS.AbstractGenericModel
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

        @test all(isone(p.trajectory.x) for p in chains_pg)
        @test mean(x.logevidence for x in chains_pg) ≈ -2 * log(2) atol = 0.01
    end

    @testset "Replay reference" begin
        mutable struct DummyModel <: AdvancedPS.AbstractGenericModel
            a::Float64
            b::Float64

            DummyModel() = new()
        end

        function (m::DummyModel)(rng)
            m.a = rand(rng, Normal())
            AdvancedPS.observe(Normal(), m.a)

            m.b = rand(rng, Normal())
            return AdvancedPS.observe(Normal(), m.b)
        end

        pg = AdvancedPS.PG(1)
        first, second = sample(DummyModel(), pg, 2)

        first_model = first.trajectory
        second_model = second.trajectory

        # Single Particle - must be replaying
        @test first_model.a ≈ second_model.a
        @test first_model.b ≈ second_model.b
        @test first.logevidence ≈ second.logevidence
    end
end
