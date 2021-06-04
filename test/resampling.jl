@testset "resampling.jl" begin
    D = [0.3, 0.4, 0.3]
    num_samples = Int(1e6)
    rng = Random.GLOBAL_RNG

    resSystematic = AdvancedPS.resample_systematic(rng, D, num_samples)
    resStratified = AdvancedPS.resample_stratified(rng, D, num_samples)
    resMultinomial = AdvancedPS.resample_multinomial(rng, D, num_samples)
    resResidual = AdvancedPS.resample_residual(rng, D, num_samples)
    AdvancedPS.resample_systematic(rng, D)

    @test sum(resSystematic .== 2) ≈ (num_samples * 0.4) atol = 1e-3 * num_samples
    @test sum(resStratified .== 2) ≈ (num_samples * 0.4) atol = 1e-3 * num_samples
    @test sum(resMultinomial .== 2) ≈ (num_samples * 0.4) atol = 1e-2 * num_samples
    @test sum(resResidual .== 2) ≈ (num_samples * 0.4) atol = 1e-2 * num_samples
end
