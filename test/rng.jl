@testset "rng.jl" begin
    @testset "sample distribution" begin
        rng = AdvancedPS.TracedRNG()
        vns = rand(rng, Distributions.Normal())
        AdvancedPS.save_state!(rng)

        rand(rng, Distributions.Normal())

        AdvancedPS.load_state(rng)
        new_vns = rand(rng, Distributions.Normal())
        @test new_vns ≈ vns
    end
end
