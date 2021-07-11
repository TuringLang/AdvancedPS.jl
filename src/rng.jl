"""
Data structure to keep track of the history of the random stream
produced by RNG.
"""
mutable struct TracedRNG{T} <: Random.AbstractRNG where {T<:Random.AbstractRNG}
    count::Base.RefValue{Int}
    rng::T
    seed::Array
    states::Array{T}
end

# Set seed manually, for init ?
function Random.seed!(rng::TracedRNG, seed)
    rng.rng.seed = seed
    return Random.seed!(rng.rng, seed)
end

# Reset the rng to the initial seed
Random.seed!(rng::TracedRNG) = Random.seed!(rng.rng, rng.seed)

TracedRNG() = TracedRNG(Random.MersenneTwister()) # Pick up an explicit RNG from Random
TracedRNG(rng::Random.AbstractRNG) = TracedRNG(Ref(0), rng, rng.seed, [rng])
TracedRNG(rng::Random._GLOBAL_RNG) = TracedRNG(Random.default_rng())

# Intercept rand
# https://github.com/JuliaLang/julia/issues/30732
Random.rng_native_52(r::TracedRNG) = UInt64

function Base.rand(rng::TracedRNG, ::Type{T}) where {T}
    res = Base.rand(rng.rng, T)
    inc_count!(rng, length(res))
    push!(rng.states, copy(rng.rng))
    return res
end

inc_count!(rng::TracedRNG) = inc_count!(rng, 1)

inc_count!(rng::TracedRNG, n::Int) = rng.count[] += n

curr_count(t::TracedRNG) = t.count[]
