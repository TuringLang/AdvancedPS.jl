module AdvancedPS

using AbstractMCMC: AbstractMCMC
using Distributions: Distributions
using Random: Random
using StatsFuns: StatsFuns
using Random123: Random123

""" Abstract type for an abstract model formulated in the state space form
"""
abstract type AbstractStateSpaceModel <: AbstractMCMC.AbstractModel end

include("resampling.jl")
include("rng.jl")
include("model.jl")
include("container.jl")
include("pgas.jl")
include("smc.jl")

if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Libtask = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f" include("../ext/LibtaskExt.jl")
    end
end

end
