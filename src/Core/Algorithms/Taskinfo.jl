

# The idea behind the TaskInfo struct is that it
# allows to propagate information trough the computation.
# This is important because we do not want to

mutable struct SMCTaskInfo{T} <: AbstractTaskInfo where {T <:AbstractFloat}
    # This corresponds to p(yₜ | xₜ) p(xₜ | xₜ₋₁) / γ(xₜ | xₜ₋₁, yₜ)
    # where γ is the porposal.
    # We need this variable to compute the weights!
    logp::T
    # This corresponds to p(xₜ | xₜ₋₁) p(xₜ₋₁ | xₜ₋₂) ⋯ p(x₀)
    # or |x_{0:t-1} for non markovian models, we need this to compute
    # the ancestor weights.
    logpseq::T
end

SMCTaskInfo() = SMCTaskInfo(0.0, 0.0)

const PGTaskInfo = SMCTaskInfo



mutable struct PGASTaskInfo{T, F} <: AbstractTaskInfo where {T <: AbstractFloat}
    # Same as above
    logp::T
    logpseq::T
    # The ancestor weight
    ancestor_weight::Float64
end
function PGASTaskInfo(logp::Float64,logpseq::Float64)
    PGASTaskInfo(logp, logpseq, 0.0)
end

function Base.copy(info::PGTaskInfo)
    PGTaskInfo(info.logp, info.logpseq)
end
function Base.copy(info::PGASTaskInfo)
     PGASTaskInfo(info.logp, info.logpseq, info.ancestor_weight)
end

reset_logp!(ti::AbstractTaskInfo) = (ti.logp = 0.0)
set_ancestor_weight!(ti::PGASTaskInfo, w::Float64) = (ti.ancestor_weight = w)

@inline function reset_task_info!(ti::PGASTaskInfo)
    ti.logp = 0.0
    ti.logpseq = 0.0
    ti.ancestor_weight = 0.0
end

@inline function reset_task_info!(ti::PGTaskInfo)
    ti.logp = 0.0
    ti.logpseq = 0.0
end
