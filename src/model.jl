"""
	Trace{F,R}
"""
mutable struct Trace{F,R}
    model::F
    rng::R
end

const Particle = Trace
const SSMTrace{R} = Trace{<:SSMProblems.AbstractStateSpaceModel,R}
const GenericTrace{R} = Trace{<:AbstractGenericModel,R}

mutable struct TracedSSM{SSM,XT,YT} <: SSMProblems.AbstractStateSpaceModel
    model::SSM
    X::Vector{XT}
    Y::Vector{YT}

    function TracedSSM(
        model::SSMProblems.StateSpaceModel{T,LD,OP}, Y::Vector{YT}
    ) where {T,LD,OP,YT}
        XT = eltype(LD)
        @assert eltype(OP) == YT
        return new{SSMProblems.StateSpaceModel{T,LD,OP},XT,YT}(model, Vector{XT}(), Y)
    end
end

(model::SSMProblems.StateSpaceModel)(Y::AbstractVector) = TracedSSM(model, Y)
dynamics(ssm::TracedSSM, step::Int) = ssm.model.dyn
observation(ssm::TracedSSM, step::Int) = ssm.model.obs

isdone(ssm::TracedSSM, step::Int) = step > length(ssm.Y)

reset_logprob!(::AdvancedPS.Particle) = nothing
reset_model(f) = deepcopy(f)
delete_retained!(f) = nothing

"""
    isdone(model::SSMProblems.AbstractStateSpaceModel, step)

Returns `true` if we reached the end of the model execution
"""
function isdone end

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

# We need this one to be visible outside of the extension for dispatching (Turing.jl).
struct LibtaskModel{F,T}
    f::F
    ctask::T
end
