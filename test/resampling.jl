using Test
using AdvancedPS

@testset "resampling.jl" begin
    D = [0.3, 0.4, 0.3]
    num_samples = Int(1e6)
    resSystematic = resample_systematic(D, num_samples )
    resStratified = resample_stratified(D, num_samples )
    resMultinomial= resample_multinomial(D, num_samples )
    resResidual   = resample_residual(D, num_samples )

    @test count(==(2), resSystematic) ≈ 0.4 * num_samples atol=1e-3*num_samples
    @test count(==(2), resStratified) ≈ 0.4 * num_samples atol=1e-3*num_samples
    @test count(==(2), resMultinomial) ≈ 0.4 * num_samples atol=1e-2*num_samples
    @test count(==(2), resResidual) ≈ 0.4 * num_samples atol=1e-2*num_samples
end
