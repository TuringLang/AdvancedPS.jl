module AdvancedPS

using AbstractMCMC: AbstractMCMC
using Distributions: Distributions
using Random: Random
using StatsFuns: StatsFuns
using Random123: Random123
using SSMProblems: SSMProblems

abstract type AbstractParticleModel <: AbstractMCMC.AbstractModel end

abstract type AbstractParticleSampler <: AbstractMCMC.AbstractSampler end

""" Abstract type for an abstract model formulated in the state space form
"""
abstract type AbstractStateSpaceModel <: AbstractParticleModel end
abstract type AbstractGenericModel <: AbstractParticleModel end

# TODO(penelopeysm): This should be upstreamed to Turing together with anything that is
# Turing-specific in LibtaskExt.
abstract type AbstractTuringLibtaskModel <: AbstractGenericModel end

include("resampling.jl")
include("rng.jl")
include("model.jl")
include("container.jl")
include("smc.jl")
include("pgas.jl")

end
