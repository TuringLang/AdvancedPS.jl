"""
    Trace{F,R}
"""
mutable struct Trace{F,R}
    model::F
    rng::R
end

const Particle = Trace
const SSMTrace{R} = Trace{<:AbstractStateSpaceModel,R}

# reset log probability
reset_logprob!(::AdvancedPS.Particle) = nothing

reset_model(f) = deepcopy(f)
delete_retained!(f) = nothing

Base.copy(trace::Trace) = Trace(copy(trace.model), deepcopy(trace.rng))

function observe end

"""
    gen_refseed!(particle::Particle)

Generate a new seed for the reference particle
"""
function gen_refseed!(particle::Particle)
    seed = split(state(particle.rng.rng), 1)
    return safe_set_refseed!(particle.rng, seed[1])
end
