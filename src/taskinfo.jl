

# The idea behind the TaskInfo struct is that it
# allows to propagate information trough the computation.
# This is important because we do not want to

mutable struct PGTaskInfo <: AbstractTaskInfo
    # This corresponds to p(y_t | x_t)*p(x_t|x_t-1)/ γ(x_t|x_t-1,y_t),
    # where γ is the porposal.
    # We need this variable to compute the weights!
    logp::Float64

    # This corresponds to p(x_t|x_t-1)*p(x_t-1|x_t-2)*... *p(x_0)
    # or |x_{0:t-1} for non markovian models, we need this to compute
    # the ancestor weights.

    logpseq::Float64
end

function copy(info::PGTaskInfo)
    PGTaskInfo(info.logp, info.logpseq)
end
