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
        resampler = AdvancedPS.ResampleWithESSThreshold(1.0)

        part = particles[3]
        AdvancedPS.advance!(part)
        AdvancedPS.advance!(part)
        AdvancedPS.advance!(part)

        ref = AdvancedPS.forkr(part)
        particles[3] = ref
        pc = AdvancedPS.ParticleContainer(particles, AdvancedPS.TracedRNG(), base_rng)

        AdvancedPS.reweight!(pc, ref)
        AdvancedPS.resample_propagate!(base_rng, pc, resampler, ref)

        AdvancedPS.reweight!(pc, ref)
        pc.logWs = [-Inf, 0, -Inf] # Force ancestor update to second particle
        AdvancedPS.resample_propagate!(base_rng, pc, resampler, ref)

        AdvancedPS.reweight!(pc, ref)
        @test all(pc.vals[2].model.X[1:2] .≈ ref.model.X[1:2])

        terminal_values = [part.model.X[3] for part in pc.vals]
        @test length(Set(terminal_values)) == 3 # All distinct
    end
end
