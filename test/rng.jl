@testset "rng.jl" begin
    @testset "sample distribution" begin
        rng = AdvancedPS.TracedRNG()
        vns = rand(rng, Distributions.Normal())
        AdvancedPS.save_state!(rng)

        rand(rng, Distributions.Normal())

        AdvancedPS.load_state!(rng)
        new_vns = rand(rng, Distributions.Normal())
        @test new_vns ≈ vns
    end

    @testset "split" begin
        rng = AdvancedPS.TracedRNG()
        key = rng.rng.key
        new_key, = AdvancedPS.split(key, 1)

        @test key ≠ new_key

        Random.seed!(rng, new_key)
        @test rng.rng.key === new_key
    end
end
