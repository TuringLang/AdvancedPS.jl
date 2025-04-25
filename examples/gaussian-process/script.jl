using Pkg
Pkg.activate("/sftwr/user-pkg/m1cak00/julia/dev/AdvancedPS/examples/gaussian-process")

# # Gaussian Process State-Space Model (GP-SSM)
using LinearAlgebra
using Random
using AdvancedPS
using AbstractGPs
using Plots
using Distributions
using Libtask
using SSMProblems

# Gaussian process encoded transition dynamics
mutable struct GaussianProcessDynamics{T<:Real} <: SSMProblems.LatentDynamics{T,T}
    proc::AbstractGPs.AbstractGP
    q::T
    function GaussianProcessDynamics(q::T, kernel::KT) where {T<:Real,KT<:Kernel}
        return new{T}(GP(ZeroMean{T}(), kernel), q)
    end
end

function SSMProblems.distribution(dyn::GaussianProcessDynamics{T}) where {T<:Real}
    return Normal(zero(T), dyn.q)
end

# TODO: broken...
function SSMProblems.simulate(
    rng::AbstractRNG, dyn::GaussianProcessDynamics, step::Int, state
)
    dyn.proc = posterior(dyn.proc(step:step, 0.1), [state])
    μ, σ = mean_and_cov(dyn.proc, [step])
    return rand(rng, Normal(μ[1], sqrt(σ[1])))
end

function SSMProblems.logdensity(dyn::GaussianProcessDynamics, step::Int, state, prev_state)
    μ, σ = mean_and_cov(dyn.proc, [step])
    return logpdf(Normal(μ[1], sqrt(σ[1])), state)
end

# Linear Gaussian dynamics used for simulation
struct LinearGaussianDynamics{T<:Real} <: SSMProblems.LatentDynamics{T,T}
    a::T
    q::T
end

function SSMProblems.distribution(dyn::LinearGaussianDynamics{T}) where {T<:Real}
    return Normal(zero(T), dyn.q)
end

function SSMProblems.distribution(dyn::LinearGaussianDynamics, ::Int, state)
    return Normal(dyn.a * state, dyn.q)
end

# Observation process used in both variants of the model
struct StochasticVolatility{T<:Real} <: SSMProblems.ObservationProcess{T,T} end

function SSMProblems.distribution(::StochasticVolatility{T}, ::Int, state) where {T<:Real}
    return Normal(zero(T), exp((1 / 2) * state))
end

# Baseline model (for simulation)
function LinearGaussianStochasticVolatilityModel(a::T, q::T) where {T<:Real}
    dyn = LinearGaussianDynamics(a, q)
    obs = StochasticVolatility{T}()
    return SSMProblems.StateSpaceModel(dyn, obs)
end

# Gaussian process model (for sampling)
function GaussianProcessStateSpaceModel(q::T, kernel::KT) where {T<:Real,KT<:Kernel}
    dyn = GaussianProcessDynamics(q, kernel)
    obs = StochasticVolatility{T}()
    return SSMProblems.StateSpaceModel(dyn, obs)
end

# Everything is now ready to simulate some data. 
rng = Random.MersenneTwister(1234)
true_model = LinearGaussianStochasticVolatilityModel(0.9, 0.5)
_, x, y = sample(rng, true_model, 100);

# Create the model and run the sampler
gpssm = GaussianProcessStateSpaceModel(0.5, SqExponentialKernel())
model = gpssm(y)
pg = AdvancedPS.PGAS(20)
chains = sample(rng, model, pg, 50)
#md nothing #hide

particles = hcat([chain.trajectory.model.X for chain in chains]...)
mean_trajectory = mean(particles; dims=2);
#md nothing #hide

scatter(particles; label=false, opacity=0.01, color=:black, xlabel="t", ylabel="state")
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)
