

# The idea behind the TaskInfo struct is that it
# allows to propagate information trough the computation.
# This is important because we do not want to

mutable struct PGTaskInfo{T} <: AbstractTaskInfo where {T <:AbstractFloat}
    # This corresponds to p(yₜ | xₜ) p(xₜ | xₜ₋₁) / γ(xₜ | xₜ₋₁, yₜ)
    # where γ is the porposal.
    # We need this variable to compute the weights!
    logp::T
    # This corresponds to p(xₜ | xₜ₋₁) p(xₜ₋₁ | xₜ₋₂) ⋯ p(x₀)
    # or |x_{0:t-1} for non markovian models, we need this to compute
    # the ancestor weights.
    logpseq::T
end



mutable struct PGASTaskInfo{T} <: AbstractTaskInfo where {T <: AbstractFloat}
    # Same as above
    logp::T
    logpseq::T
    # This is important for ancesotr sampling to synchronize.
    hold::Bool
end
function PGASTaskInfo(logp::Float64,logpseq::Float64)
    PGASTaskInfo(logp,logpseq,false)
end

function Base.copy(info::PGTaskInfo) = (PGTaskInfo(info.logp, info.logpseq)
function Base.copy(info::PGASTaskInfo) = (PGASTaskInfo(info.logp, info.logpseq,info.hold)

reset_logp!(ti::AbstractTaskInfo) = (ti.logp = 0.0)
