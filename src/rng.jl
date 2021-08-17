using Random123
using Random
using Distributions

import Base.rand
import Random.seed!

# Use Philox2x for now
BASE_RNG = Philox2x

"""
    TracedRNG{R,T}

Wrapped random number generator from Random123 to keep track of random streams during model evaluation
"""
mutable struct TracedRNG{T} <:
               Random.AbstractRNG where {T<:(Random123.AbstractR123{R} where {R})}
    count::Int
    rng::T
    keys
    counters
end

function TracedRNG(r::Random123.AbstractR123)
    return TracedRNG(1, r, typeof(r.key)[], typeof(r.ctr1)[])
end

"""
    TracedRNG()

Create a default TracedRNG
"""
function TracedRNG()
    r = BASE_RNG()
    return TracedRNG(r)
end

# Plug into Random
Random.rng_native_52(rng::TracedRNG{U}) where {U} = Random.rng_native_52(rng.rng)
Base.rand(rng::TracedRNG{U}, ::Type{T}) where {U,T} = Base.rand(rng.rng, T)

"""
    split(r::TracedRNG, n::Integer)

Split keys of the internal Philox2x into n distinct seeds
"""
function split(r::TracedRNG{T}, n::Integer) where {T}
    n == 1 && return [r.rng.key]
    return map(i -> hash(r.rng.key, r.rng.ctr1 + i), 1:n)
end

"""
    update_rng!(r::TracedRNG, seed::Number)

Set the key of the wrapped Philox2x rng
"""
function seed!(r::TracedRNG{T}, seed) where {T}
    return seed!(r.rng, seed)
end

"""
    reset_rng(r::TracedRNG, seed)

Reset the rng to the running model step
"""
function reset_rng!(rng::TracedRNG{T}) where {T}
    key = rng.keys[rng.count]
    ctr = rng.counters[rng.count]
    Random.seed!(rng.rng, key)
    return set_counter!(rng.rng, ctr)
end

function save_state!(r::TracedRNG{T}) where {T}
    push!(r.keys, r.rng.key)
    return push!(r.counters, r.rng.ctr1)
end

Base.copy(r::TracedRNG{T}) where {T} = TracedRNG(r.count, copy(r.rng), copy(r.keys))

"""
    set_count!(r::TracedRNG, n::Integer)

Set the counter of the TracedRNG, used to keep track of the current model step
"""
set_count!(r::TracedRNG, n::Integer) = r.count = n

inc_count!(r::TracedRNG, n::Integer) = r.count += n
inc_count!(r::TracedRNG) = inc_count!(r, 1)
