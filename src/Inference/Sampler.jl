
abstract type AbstractPFSampler <: AbstractSampler end

# The particle container is our state,

struct SMCSampler{PC, ALG, UF, C} <: AbstractPFSampler where {PC<:ParticleContainer, ALG<:SMCAlgorithm, UF<:SMCUtilityFunctions}
    pc        :: PC
    alg       :: ALG
    uf        :: UF
    vi        :: C
end


function Sampler(alg:: ALG, uf::UF, vi::T) where {
    ALG<: SMCAlgorithm,
    UF<: SMCUtilityFunctions,
}
    pc = ParticleContainer(Trace{typeof(vi),SMCTaskInfo}[])
    SMCSampler(pc, alg, uf, uf.empty!(vi))
end




struct PGSampler{T, ALG, UF, C} <: AbstractPFSampler where {
    T <:Particle,
    ALG<:SMCAlgorithm,
    UF<:AbstractUtilityFunctions
}

    alg       :: ALG
    uf        :: UF
    ref_traj  :: Union{T, Nothing}
    vi        :: C
end


Sampler(alg:: ALG, uf::UF, vi::T) where {
    ALG<: PGAlgorithm,
    UF<: AbstractUtilityFunctions,
    M <:AbstractPFModel
} = PGSampler(pc, alg, uf, nothing)
