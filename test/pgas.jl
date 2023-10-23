@testset "pgas.jl" begin
    mutable struct Params
        a::Float64
        q::Float64
        r::Float64
    end

    mutable struct BaseModel <: AdvancedPS.AbstractStateSpaceModel
        X::Vector{Float64}
        θ::Params
        BaseModel(params::Params) = new(Vector{Float64}(), params)
    end

    AdvancedPS.initialization(model::BaseModel) = Normal(0, model.θ.q)
    function AdvancedPS.transition(model::BaseModel, state, step)
        return Distributions.Normal(model.θ.a * state, model.θ.q)
    end
    function AdvancedPS.observation(model::BaseModel, state, step)
        return Distributions.logpdf(Distributions.Normal(state, model.θ.r), 0)
    end

    AdvancedPS.isdone(::BaseModel, step) = step > 3

    @testset "fork reference" begin
        model = BaseModel(Params(0.9, 0.32, 1))
        part = AdvancedPS.Trace(model, AdvancedPS.TracedRNG())

        AdvancedPS.advance!(part)
        AdvancedPS.advance!(part)
        AdvancedPS.advance!(part)
        @test length(part.model.X) == 3

        trajectory = deepcopy(part.model.X)
        ref = AdvancedPS.forkr(part)
        @test all(trajectory .≈ ref.model.X)

        AdvancedPS.advance!(ref)
        new_part = AdvancedPS.fork(ref, true)
        @test length(new_part.model.X) == 1
        @test new_part.model.X[1] ≈ ref.model.X[1]
    end

    @testset "update reference" begin
        base_rng = Random.MersenneTwister(31)
        particles = [
            AdvancedPS.Trace(BaseModel(Params(0.9, 0.31, 1)), AdvancedPS.TracedRNG()) for
            _ in 1:3
        ]
        sampler = AdvancedPS.PGAS(3)
        resampler = AdvancedPS.ResampleWithESSThreshold(1.0)

        part = particles[3]
        AdvancedPS.advance!(part)
        AdvancedPS.advance!(part)
        AdvancedPS.advance!(part)

        ref = AdvancedPS.forkr(part)
        particles[3] = ref
        pc = AdvancedPS.ParticleContainer(particles, AdvancedPS.TracedRNG(), base_rng)

        AdvancedPS.reweight!(pc, ref)
        AdvancedPS.resample_propagate!(base_rng, pc, sampler, resampler, ref)

        AdvancedPS.reweight!(pc, ref)
        pc.logWs = [-Inf, 0, -Inf] # Force ancestor update to second particle
        AdvancedPS.resample_propagate!(base_rng, pc, sampler, resampler, ref)

        AdvancedPS.reweight!(pc, ref)
        @test all(pc.vals[2].model.X[1:2] .≈ ref.model.X[1:2])

        terminal_values = [part.model.X[3] for part in pc.vals]
        @test length(Set(terminal_values)) == 3 # All distinct
    end

    @testset "constructor" begin
        sampler = AdvancedPS.PGAS(10)
        @test sampler.nparticles == 10
        @test sampler.resampler === AdvancedPS.ResampleWithESSThreshold(1.0) # adaptive PG-AS ?
    end

    @testset "rng stability" begin
        model = BaseModel(Params(0.9, 0.32, 1))
        seed = 10
        rng = Random.MersenneTwister(seed)

        for sampler in [AdvancedPS.PGAS(10)]
            Random.seed!(rng, seed)
            chain1 = sample(rng, model, sampler, 10)
            vals1 = hcat([chain.trajectory.model.X for chain in chain1]...)

            Random.seed!(rng, seed)
            chain2 = sample(rng, model, sampler, 10)
            vals2 = hcat([chain.trajectory.model.X for chain in chain2]...) # TODO: Create proper chains

            @test vals1 ≈ vals2
        end

        # Test stability for SMC
        sampler = AdvancedPS.SMC(10)
        Random.seed!(rng, seed)
        chain1 = sample(rng, model, sampler)
        vals1 = hcat([trace.model.X for trace in chain1.trajectories]...)

        Random.seed!(rng, seed)
        chain2 = sample(rng, model, sampler)
        vals2 = hcat([trace.model.X for trace in chain2.trajectories]...)

        @test vals1 ≈ vals2
    end

    # Smoke test mostly
    @testset "smc sampler" begin
        model = BaseModel(Params(0.9, 0.32, 1))
        npart = 10

        sampler = AdvancedPS.SMC(npart)
        chains = sample(model, sampler)

        @test length(chains.trajectories) == npart
        @test length(chains.trajectories[1].model.X) == 3
    end
end
