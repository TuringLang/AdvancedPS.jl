


struct PFTransition{T, F<:AbstractFloat} <: AbstractPFTransition
    θ::T
    lp::F
    le::F
    weight::Vector{F}
end

transition_type(spl::Sampler{<:ParticleInference}) = ParticleTransition
function additional_parameters(::Type{<:ParticleTransition})
    return [:lp,:le, :weight]
end
