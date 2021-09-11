# Default RNG type for when nothing is specified
const _BASE_RNG = Random123.Philox2x

"""
    TracedRNG{R,N,T}

Wrapped random number generator from Random123 to keep track of random streams during model evaluation
"""
mutable struct TracedRNG{R,N,T<:Random123.AbstractR123{R}} <: Random.AbstractRNG
    "Model step counter"
    count::Int
    "Inner RNG"
    rng::T
    "Array of keys"
    keys::Array{R,N}
end

"""
    TracedRNG(r::Random123.AbstractR123=AdvancedPS._BASE_RNG())
Create a `TracedRNG` with `r` as the inner RNG. 
"""
function TracedRNG(r::Random123.AbstractR123=_BASE_RNG())
    Random123.set_counter!(r, 0)
    return TracedRNG(1, r, typeof(r.key)[])
end

# Connect to the Random API
Random.rng_native_52(rng::TracedRNG) = Random.rng_native_52(rng.rng)
Base.rand(rng::TracedRNG, ::Type{T}) where {T} = Base.rand(rng.rng, T)

"""
    split(key::Integer, n::Integer=1)

Split `key` into `n` new keys
"""
function split(key::Integer, n::Integer=1)
    return [hash(key, i) for i in UInt(1):UInt(n)]
end

"""
    load_state!(r::TracedRNG)

Load state from current model iteration. Random streams are now replayed
"""
function load_state!(rng::TracedRNG)
    key = rng.keys[rng.count]
    Random.seed!(rng.rng, key)
    return Random123.set_counter!(rng.rng, 0)
end

"""
    update_rng!(rng::TracedRNG)

Set key and counter of inner rng in `rng` to `key` and the running model step to 0
"""
function Random.seed!(rng::TracedRNG, key)
    Random.seed!(rng.rng, key)
    return Random123.set_counter!(rng.rng, 0)
end

""" 
    save_state!(r::TracedRNG)

Add current key of the inner rng in `r` to `keys`.
"""
function save_state!(r::TracedRNG)
    return push!(r.keys, r.rng.key)
end

Base.copy(r::TracedRNG) = TracedRNG(r.count, copy(r.rng), deepcopy(r.keys))

"""
    set_counter!(r::TracedRNG, n::Integer)

Set the counter of the inner rng in `r`, used to keep track of the current model step
"""
Random123.set_counter!(r::TracedRNG, n::Integer) = r.count = n

"""
    inc_counter!(r::TracedRNG, n::Integer=1)

Increase the model step counter by `n`
"""
inc_counter!(r::TracedRNG, n::Integer=1) = r.count += n
