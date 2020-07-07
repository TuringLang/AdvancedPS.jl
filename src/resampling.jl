# 2 dichotomies between resamplers: 
#     - whether or not to use ESS thresholds
#     - whether to use systematic, residual, multinomial, or stratified resampling schemes

#############################################
## Resample only when ESS ≤ a preset value ##
#############################################


# modifies dispatch in resample_propagate!
struct ResampleWithESSThreshold{R, T<:Real}
    scheme::R
    threshold::T
end

function ResampleWithESSThreshold(scheme = resample_systematic)
    ResampleWithESSThreshold(scheme, 0.5)
end

#############################################
## Resampling schemes for particle filters ##
#############################################

# Some references
#  - http://arxiv.org/pdf/1301.4019.pdf
#  - http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
# Code adapted from: http://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering

# More stable, faster version of rand(Categorical)
function randcat(p::AbstractVector{<:Real})
    T = eltype(p)
    r = rand(T)
    s = 1
    for j in eachindex(p)
        r -= p[j]
        if r <= zero(T)
            s = j
            break
        end
    end
    return s
end

"""
    resample_multinomial(w, num_particles)
Multinomial resampling scheme
"""
function resample_multinomial(w::AbstractVector{<:Real}, num_particles::Integer)
    return rand(Distributions.sampler(Categorical(w)), num_particles)
end

"""
    resample_residual(w, num_particles)
Residual resampling scheme
"""
function resample_residual(w::AbstractVector{<:Real}, num_particles::Integer)

    M = length(w)

    # "Repetition counts" (plus the random part, later on):
    Ns = floor.(length(w) .* w)

    # The "remainder" or "residual" count:
    R = Int(sum(Ns))

    # The number of particles which will be drawn stocastically:
    M_rdn = num_particles - R

    # The modified weights:
    Ws = (M .* w - floor.(M .* w)) / M_rdn

    # Draw the deterministic part:
    indx1, i = Array{Int}(undef, R), 1
    for j in 1:M
        for k in 1:Ns[j]
            indx1[i] = j
            i += 1
        end
    end

    # And now draw the stochastic (Multinomial) part:
    return append!(indx1, rand(Distributions.sampler(Categorical(w)), M_rdn))
end

"""
    resample_stratified(weights, n)
Return a vector of `n` samples `x₁`, ..., `xₙ` from the numbers 1, ..., `length(weights)`,
generated by stratified resampling.
In stratified resampling `n` ordered random numbers `u₁`, ..., `uₙ` are generated, where
``uₖ \\sim U[(k - 1) / n, k / n)``. Based on these numbers the samples `x₁`, ..., `xₙ`
are selected according to the multinomial distribution defined by the normalized `weights`,
i.e., `xᵢ = j` if and only if
``uᵢ \\in [\\sum_{s=1}^{j-1} weights_{s}, \\sum_{s=1}^{j} weights_{s})``.
"""
function resample_stratified(weights::AbstractVector{<:Real}, n::Integer)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]

    # generate all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # sample next `u` (scaled by `n`)
        u = oftype(v, i - 1 + rand())

        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")

            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end

        # save the next sample
        samples[i] = sample
    end

    return samples
end

"""
    resample_systematic(weights, n)
Return a vector of `n` samples `x₁`, ..., `xₙ` from the numbers 1, ..., `length(weights)`,
generated by systematic resampling.
In systematic resampling a random number ``u \\sim U[0, 1)`` is used to generate `n` ordered
numbers `u₁`, ..., `uₙ` where ``uₖ = (u + k − 1) / n``. Based on these numbers the samples
`x₁`, ..., `xₙ` are selected according to the multinomial distribution defined by the
normalized `weights`, i.e., `xᵢ = j` if and only if
``uᵢ \\in [\\sum_{s=1}^{j-1} weights_{s}, \\sum_{s=1}^{j} weights_{s})``.
"""
function resample_systematic(weights::AbstractVector{<:Real}, n::Integer)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]
    u = oftype(v, rand())

    # find all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")

            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end

        # save the next sample
        samples[i] = sample

        # update `u`
        u += one(u)
    end

    return samples
end
