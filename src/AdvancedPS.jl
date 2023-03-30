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

end
