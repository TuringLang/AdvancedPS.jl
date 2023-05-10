"""
    Trace{F,R}
"""
mutable struct Trace{F,R}
    model::F
    rng::R

    Trace(model::F, rng::R) where {F,R} = new{F,R}(model, rng)
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
    delete_seeds!(particle::Particle)

Truncate the seed history from the `particle` rng. When forking the reference Particle
we need to keep the seeds up to the current model iteration but generate new seeds
and random values afterward.
"""
function delete_seeds!(particle::Particle)
    return particle.rng.keys = particle.rng.keys[1:(particle.rng.count - 1)]
end

"""
    gen_refseed!(particle::Particle)

Generate a new seed for the reference particle
"""
function gen_refseed!(particle::Particle)
    seed = split(state(particle.rng.rng), 1)
    return safe_set_refseed!(particle.rng, seed[1])
end
