using AdvancedPS
using Test

@testset "particlecontainer.jl" begin
    @testset "copy particle container" begin
        pc = ParticleContainer(Trace[])
        newpc = copy(pc)

        @test newpc.logWs == pc.logWs
        @test typeof(pc) === typeof(newpc)
    end
end