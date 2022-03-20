using AdvancedPS
using AbstractMCMC
using Distributions
using Libtask
using Random
using Test

@testset "AdvancedPS.jl" begin
    @testset "Resampling tests" begin
        include("resampling.jl")
    end
    @testset "Container tests" begin
        include("container.jl")
    end
    @testset "SMC and PG tests" begin
        include("smc.jl")
    end
    @testset "RNG tests" begin
        include("rng.jl")
    end
    @testset "PG-AS" begin
        include("pgas.jl")
    end
end
