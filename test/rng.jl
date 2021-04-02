@testset "rng.jl" begin
    @testset "sample distribution" begin
        rng = AdvancedPS.TracedRNG()
        vns = rand!(rng, Distributions.Normal())

        @test AdvancedPS.curr_count(rng) === 1

        rand!(rng, Distributions.Normal())
        Random.seed!(rng)
        new_vns = rand!(rng, Distributions.Normal())
        @test new_vns ≈ vns
    end

    @testset "inc count" begin
        rng = AdvancedPS.TracedRNG()
        AdvancedPS.inc_count!(rng)
        @test AdvancedPS.curr_count(rng) == 1

        AdvancedPS.inc_count!(rng, 2)
        @test AdvancedPS.curr_count(rng) == 3
    end

    @testset "curr count" begin
        rng = AdvancedPS.TracedRNG()
        @test AdvancedPS.curr_count(rng) == 0
    end
end
