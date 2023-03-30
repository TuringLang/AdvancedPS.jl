"""
    Trace{F,R}
"""
mutable struct Trace{F,R}
    model::F
    rng::R
end

const Particle = Trace
const SSMTrace{R} = Trace{<:AbstractStateSpaceModel,R}
