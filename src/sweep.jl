"""
resample_propagate!(pc::ParticleContainer[, randcat = resample_systematic, ref = nothing;
                    weights = getweights(pc)])
Resample and propagate the particles in `pc`.
Function `randcat` is used for sampling ancestor indices from the categorical distribution
of the particle `weights`. For Particle Gibbs sampling, one can provide a reference particle
`ref` that is ensured to survive the resampling step.
"""
function resample_propagate!(
pc::ParticleContainer,
randcat = resample_systematic,
ref::Union{Particle, Nothing} = nothing;
weights = getweights(pc)
)
# check that weights are not NaN
@assert !any(isnan, weights)

# sample ancestor indices
n = length(pc)
nresamples = ref === nothing ? n : n - 1
indx = randcat(weights, nresamples)

# count number of children for each particle
num_children = zeros(Int, n)
@inbounds for i in indx
    num_children[i] += 1
end

# fork particles
particles = collect(pc)
children = similar(particles)
j = 0
@inbounds for i in 1:n
    ni = num_children[i]

    if ni > 0
        # fork first child
        pi = particles[i]
        isref = pi === ref
        p = isref ? fork(pi, isref) : pi
        children[j += 1] = p

        # fork additional children
        for _ in 2:ni
            children[j += 1] = fork(p, isref)
        end
    end
end

if ref !== nothing
    # Insert the retained particle. This is based on the replaying trick for efficiency
    # reasons. If we implement PG using task copying, we need to store Nx * T particles!
    @inbounds children[n] = ref
end

# replace particles and log weights in the container with new particles and weights
pc.vals = children
reset_logweights!(pc)

pc
end



function resample_propagate!(
pc::ParticleContainer,
resampler::ResampleWithESSThreshold,
ref::Union{Particle,Nothing} = nothing;
weights = getweights(pc)
)
# Compute the effective sample size ``1 / ∑ wᵢ²`` with normalized weights ``wᵢ``
ess = inv(sum(abs2, weights))

if ess ≤ resampler.threshold * length(pc)
    resample_propagate!(pc, resampler.resampler, ref; weights = weights)
end

pc
end


"""
reweight!(pc::ParticleContainer)
Check if the final time step is reached, and otherwise reweight the particles by
considering the next observation.
"""
function reweight!(pc::ParticleContainer)
n = length(pc)

particles = collect(pc)
numdone = 0
for i in 1:n
    p = particles[i]

    # Obtain ``\\log p(yₜ | y₁, …, yₜ₋₁, x₁, …, xₜ, θ₁, …, θₜ)``, or `nothing` if the
    # the execution of the model is finished.
    # Here ``yᵢ`` are observations, ``xᵢ`` variables of the particle filter, and
    # ``θᵢ`` are variables of other samplers.
    score = Libtask.consume(p)

    if score === nothing
        numdone += 1
    else
        # Increase the unnormalized logarithmic weights, accounting for the variables
        # of other samplers.
        increase_logweight!(pc, i, score + getlogp(p.vi))

        # Reset the accumulator of the log probability in the model so that we can
        # accumulate log probabilities of variables of other samplers until the next
        # observation.
        resetlogp!(p.vi)
    end
end

# Check if all particles are propagated to the final time point.
numdone == n && return true

# The posterior for models with random number of observations is not well-defined.
if numdone != 0
    error("mis-aligned execution traces: # particles = ", n,
          " # completed trajectories = ", numdone,
          ". Please make sure the number of observations is NOT random.")
end

return false
end

"""
sweep!(pc::ParticleContainer, resampler)
Perform a particle sweep and return an unbiased estimate of the log evidence.
The resampling steps use the given `resampler`.
# Reference
Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential monte carlo samplers.
Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(3), 411-436.
"""
function sweep!(pc::ParticleContainer, resampler)
# Initial step:

# Resample and propagate particles.
resample_propagate!(pc, resampler)

# Compute the current normalizing constant ``Z₀`` of the unnormalized logarithmic
# weights.
# Usually it is equal to the number of particles in the beginning but this
# implementation covers also the unlikely case of a particle container that is
# initialized with non-zero logarithmic weights.
logZ0 = logZ(pc)

# Reweight the particles by including the first observation ``y₁``.
isdone = reweight!(pc)

# Compute the normalizing constant ``Z₁`` after reweighting.
logZ1 = logZ(pc)

# Compute the estimate of the log evidence ``\\log p(y₁)``.
logevidence = logZ1 - logZ0

# For observations ``y₂, …, yₜ``:
while !isdone
    # Resample and propagate particles.
    resample_propagate!(pc, resampler)

    # Compute the current normalizing constant ``Z₀`` of the unnormalized logarithmic
    # weights.
    logZ0 = logZ(pc)

    # Reweight the particles by including the next observation ``yₜ``.
    isdone = reweight!(pc)

    # Compute the normalizing constant ``Z₁`` after reweighting.
    logZ1 = logZ(pc)

    # Compute the estimate of the log evidence ``\\log p(y₁, …, yₜ)``.
    logevidence += logZ1 - logZ0
end

return logevidence
end
