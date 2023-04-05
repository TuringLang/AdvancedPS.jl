# Default RNG type for when nothing is specified
const _BASE_RNG = Random123.Philox2x # Rng with state bigger than 1 are broken because of Split

"""
    TracedRNG{R,N,T}

Wrapped random number generator from Random123 to keep track of random streams during model evaluation
"""
mutable struct TracedRNG{R,N,T<:Random123.AbstractR123} <: Random.AbstractRNG
    "Model step counter"
    count::Int
    "Inner RNG"
    rng::T
    "Array of keys"
    keys::Array{R,N}
    "Reference particle alternative seed"
    refseed::Union{R,Nothing}
end

"""
    TracedRNG(r::Random123.AbstractR123=AdvancedPS._BASE_RNG())
Create a `TracedRNG` with `r` as the inner RNG.
"""
function TracedRNG(r::Random123.AbstractR123=_BASE_RNG())
    Random123.set_counter!(r, 0)
    return TracedRNG(1, r, Random123.seed_type(r)[], nothing)
end

# Connect to the Random API
Random.rng_native_52(rng::TracedRNG) = Random.rng_native_52(rng.rng)
Base.rand(rng::TracedRNG, ::Type{T}) where {T} = Base.rand(rng.rng, T)

"""
    split(key::Integer, n::Integer=1)

Split `key` into `n` new keys
"""
function split(key::Integer, n::Integer=1)
    T = typeof(key) # Make sure the type of `key` is consistent on W32 and W64 systems.
    return T[hash(key, i) for i in UInt(1):UInt(n)]
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
    Random.seed!(rng::TracedRNG, key)

Set key and counter of inner rng in `rng` to `key` and the running model step to 0
"""
function Random.seed!(rng::TracedRNG, key)
    Random.seed!(rng.rng, key)
    return Random123.set_counter!(rng.rng, 0)
end

"""
    gen_seed(rng::Random.AbstractRNG, subrng::TracedRNG, sampler::Random.Sampler)

Generate a `seed` for the subrng based on top-level `rng` and `sampler`
"""
function gen_seed(rng::Random.AbstractRNG, ::TracedRNG{<:Integer}, sampler::Random.Sampler)
    return rand(rng, sampler)
end

function gen_seed(
    rng::Random.AbstractRNG, ::TracedRNG{<:NTuple{N}}, sampler::Random.Sampler
) where {N}
    return Tuple(rand(rng, sampler, N))
end

"""
    save_state!(r::TracedRNG)

Add current key of the inner rng in `r` to `keys`.
"""
function save_state!(r::TracedRNG)
    return push!(r.keys, state(r.rng))
end

state(rng::Random123.Philox2x) = rng.key
state(rng::Random123.Philox4x) = (rng.key1, rng.key2)

function Base.copy(rng::TracedRNG)
    return TracedRNG(rng.count, copy(rng.rng), deepcopy(rng.keys), rng.refseed)
end

# Add an extra seed to the reference particle keys array to use as an alternative stream
# (we don't need to tack this one)
#
# We have to be careful when spliting the reference particle.
# Since we don't know the seed tree from the previous SMC run we cannot reuse any of the intermediate seed
# in the TracedRNG container. We might collide with a previous seed and the children particle would collapse
# to the reference particle. A solution to solve this is to have an extra stream attached to the reference particle
# that we only use to seed the children of the reference particle.
#
safe_set_refseed!(rng::TracedRNG{R}, seed::R) where {R} = rng.refseed = seed
safe_get_refseed(rng::TracedRNG) = rng.refseed

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
