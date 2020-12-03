using AdvancedPS
using Libtask
using Test

@testset "AdvancedPS.jl" begin
    @testset "Resampling tests" begin include("resampling.jl") end
    @testset "Container tests" begin include("container.jl") end
end
