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

# Dispatch to Random, keep track of numbers generated
function Random.rand!(rng::TracedRNG, values...)
    res = Random.rand(rng.rng, values...)
    n = length(res)
    inc_count!(rng, n)
    return res
end

inc_count!(rng::TracedRNG) = inc_count!(rng, 1)

inc_count!(rng::TracedRNG, n::Int) = rng.count[] += n

curr_count(t::TracedRNG) = t.count[]
