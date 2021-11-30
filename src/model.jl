struct Trace{F,R<:TracedRNG}
    f::F
    ctask::Libtask.CTask
    rng::R
end

struct SSMTrace{F<:AdvancedPS.AbstractStateSpaceModel,R<:TracedRNG}
    f::F
    rng::R
end

const Particle = Union{Trace,SSMTrace}

"""
    observe(dist::Distribution, x)

Observe sample `x` from distribution `dist` and yield its log-likelihood value.
"""
function observe(dist::Distributions.Distribution, x)
    return Libtask.produce(Distributions.loglikelihood(dist, x))
end
