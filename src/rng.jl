using Random123
using Random
using Distributions

import Base.rand
import Random.seed!
import Random123: set_counter!

# Default RNG type for when nothing is specified
_BASE_RNG = Philox2x

"""
    TracedRNG{R,T}

Wrapped random number generator from Random123 to keep track of random streams during model evaluation
"""
mutable struct TracedRNG{T} <:
               Random.AbstractRNG where {T<:(Random123.AbstractR123{R} where {R})}
    "Model step counter"
    count::Int
    "Inner RNG"
    rng::T
    "Array of keys"
    keys
end

"""
    TracedRNG(r::Random123.AbstractR123)

Initialize TracedRNG with r as the inner RNG 
"""
function TracedRNG(r::Random123.AbstractR123)
    set_counter!(r, 0)
    return TracedRNG(1, r, typeof(r.key)[])
end

"""
    TracedRNG()

Create a default TracedRNG
"""
function TracedRNG()
    r = _BASE_RNG()
    return TracedRNG(r)
end

# Connect to the Random API
Random.rng_native_52(rng::TracedRNG{U}) where {U} = Random.rng_native_52(rng.rng)
Base.rand(rng::TracedRNG{U}, ::Type{T}) where {U,T} = Base.rand(rng.rng, T)

"""
    split(key::Integer, n::Integer)

Split key into n new keys
"""
function split(key::Integer, n::Integer=1) where {T}
    return map(i -> hash(key, convert(UInt, i)), 1:n)
end

"""
    load_state!(r::TracedRNG, seed)

Load state from current model iteration. Random streams are now replayed
"""
function load_state!(rng::TracedRNG{T}) where {T}
    key = rng.keys[rng.count]
    Random.seed!(rng.rng, key)
    return set_counter!(rng.rng, 0)
end

"""
    update_rng!(rng::TracedRNG)

Set key and counter of inner RNG to key and the running model step
"""
function seed!(rng::TracedRNG{T}, key) where {T}
    seed!(rng.rng, key)
    return set_counter!(rng.rng, 0)
end

""" 
    save_state!(r::TracedRNG)

Track current key of the inner RNG
"""
function save_state!(r::TracedRNG{T}) where {T}
    return push!(r.keys, r.rng.key)
end

Base.copy(r::TracedRNG{T}) where {T} = TracedRNG(r.count, copy(r.rng), copy(r.keys))

"""
    set_count!(r::TracedRNG, n::Integer)

Set the counter of the TracedRNG, used to keep track of the current model step
"""
set_counter!(r::TracedRNG, n::Integer) = r.count = n

"""
    inc_counter!(r::TracedRNG, n::Integer=1)

Increase the model step counter by n
"""
inc_counter!(r::TracedRNG, n::Integer=1) = r.count += n
