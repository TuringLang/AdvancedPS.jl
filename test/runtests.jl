using AdvancedPS
using Test

@testset "AdvancedPS.jl" begin
    @testset "Resampling tests" begin include("resampling.jl") end
end
