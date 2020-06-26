@testset "resample.jl" begin
    D = [0.3, 0.4, 0.3]
    num_samples = Int(1e6)
    resSystematic = resample_systematic(D, num_samples )
    resStratified = resample_stratified(D, num_samples )
    resMultinomial= resample_multinomial(D, num_samples )
    resResidual   = resample_residual(D, num_samples )
    resample(D)
    resSystematic2=resample(D, num_samples )

    @test sum(resSystematic .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resSystematic2 .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resStratified .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resMultinomial .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
    @test sum(resResidual .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
end