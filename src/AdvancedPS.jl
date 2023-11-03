module AdvancedPS

using AbstractMCMC: AbstractMCMC
using Distributions: Distributions
using Random: Random
using StatsFuns: StatsFuns
using Random123: Random123

abstract type AbstractParticleModel <: AbstractMCMC.AbstractModel end

abstract type AbstractParticleSampler <: AbstractMCMC.AbstractSampler end

""" Abstract type for an abstract model formulated in the state space form
"""
abstract type AbstractStateSpaceModel <: AbstractParticleModel end
abstract type AbstractGenericModel <: AbstractParticleModel end

include("resampling.jl")
include("rng.jl")
include("model.jl")
include("container.jl")
include("smc.jl")
include("pgas.jl")

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Libtask = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f" include(
            "../ext/AdvancedPSLibtaskExt.jl"
        )
    end
end

end
