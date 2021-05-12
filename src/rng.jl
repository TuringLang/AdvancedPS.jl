"""
Data structure to keep track of the history of the random stream
produced by RNG.
"""
struct TracedRNG{T} <: Random.AbstractRNG where T <: Random.AbstractRNG
    count::Base.RefValue{Int}
    rng::T
    seed
end

# Set seed manually, for init ?
Random.seed!(rng::TracedRNG, seed) = Random.seed!(rng.rng, seed)
# Reset the rng to the initial seed
Random.seed!(rng::TracedRNG) = Random.seed!(rng.rng, rng.seed)

TracedRNG() = TracedRNG(Random.seed!()) # Pick up an explicit RNG from Random
TracedRNG(rng::Random.AbstractRNG) = TracedRNG(Ref(0), rng, rng.seed)

# Intercept rand
# https://github.com/JuliaLang/julia/issues/30732
Random.rng_native_52(r::TracedRNG) = UInt64

function Base.rand(rng::TracedRNG, ::Type{T}) where {T}
	res = Base.rand(rng.rng, T)
	inc_count!(rng, length(res))
	return res
end

inc_count!(rng::TracedRNG) = inc_count!(rng, 1)

inc_count!(rng::TracedRNG, n::Int) = rng.count[] += n

curr_count(t::TracedRNG) = t.count[]
