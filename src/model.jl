"""
	Trace{F,R}
"""
mutable struct Trace{F,R}
    model::F
    rng::R
end

const Particle = Trace
const SSMTrace{R} = Trace{<:AbstractStateSpaceModel,R}
const GenericTrace{R} = Trace{<:AbstractGenericModel,R}

reset_logprob!(::AdvancedPS.Particle) = nothing
reset_model(f) = deepcopy(f)
delete_retained!(f) = nothing

"""
	copy(trace::Trace)

Copy a trace. The `TracedRNG` is deep-copied. The inner model is shallow-copied. 
"""
Base.copy(trace::Trace) = Trace(copy(trace.model), deepcopy(trace.rng))

"""
	gen_refseed!(particle::Particle)

Generate a new seed for the reference particle.
"""
function gen_refseed!(particle::Particle)
    seed = split(state(particle.rng.rng), 1)
    return safe_set_refseed!(particle.rng, seed[1])
end

# A few internal functions used in the Libtask extension. Since it is not possible to access objects defined
# in an extension, we just define dummy in the main module and implement them in the extension. 
function observe end
function replay end
function addreference! end

current_trace() = current_task().storage[:__trace]

# We need this one to be visible outside of the extension for dispatching (Turing.jl).
struct LibtaskModel{F,T}
    f::F
    ctask::T
end
