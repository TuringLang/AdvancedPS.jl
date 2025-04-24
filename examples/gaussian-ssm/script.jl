# # Particle Gibbs for Gaussian state-space model
using AdvancedPS
using Random
using Distributions
using Plots
using SSMProblems

# We consider the following linear state-space model with Gaussian innovations. The latent state is a simple gaussian random walk
# and the observation is linear in the latent states, namely:
# 
# ```math
#  x_{t+1} = a x_{t} + \epsilon_t \quad \epsilon_t \sim \mathcal{N}(0,q^2)
# ```
# ```math
#  y_{t} = x_{t} + \nu_{t} \quad \nu_{t} \sim \mathcal{N}(0, r^2) 
# ```
#
# Here we assume the static parameters $\theta = (a, q^2, r^2)$ are known and we are only interested in sampling from the latent states $x_t$. 
# To use particle gibbs with the ancestor sampling update step we need to provide both the transition and observation densities. 
#
# From the definition above we get: 
# ```math
#  x_{t+1} \sim f_{\theta}(x_{t+1}|x_t) = \mathcal{N}(a x_t, q^2)
# ```
# ```math
#  y_t \sim g_{\theta}(y_t|x_t) = \mathcal{N}(x_t, q^2)
# ```
# as well as the initial distribution $f_0(x) = \mathcal{N}(0, q^2/(1-a^2))$.

# To use `AdvancedPS` we first need to define a model type that subtypes `AdvancedPS.AbstractStateSpaceModel`.
mutable struct Parameters{T<:Real}
    a::T
    q::T
    r::T
end

struct LinearGaussianDynamics{T<:Real} <: SSMProblems.LatentDynamics{T,T}
    a::T
    q::T
end

function SSMProblems.distribution(dyn::LinearGaussianDynamics{T}; kwargs...) where {T<:Real}
    return Normal(zero(T), sqrt(dyn.q^2 / (1 - dyn.a^2)))
end

function SSMProblems.distribution(dyn::LinearGaussianDynamics, step::Int, state; kwargs...)
    return Normal(dyn.a * state, dyn.q)
end

struct LinearGaussianObservation{T<:Real} <: SSMProblems.ObservationProcess{T,T}
    r::T
end

function SSMProblems.distribution(
    obs::LinearGaussianObservation, step::Int, state; kwargs...
)
    return Normal(state, obs.r)
end

function LinearGaussianStateSpaceModel(θ::Parameters)
    dyn = LinearGaussianDynamics(θ.a, θ.q)
    obs = LinearGaussianObservation(θ.r)
    return SSMProblems.StateSpaceModel(dyn, obs)
end

# Everything is now ready to simulate some data. 
rng = Random.MersenneTwister(1234)
θ = Parameters(0.9, 0.32, 1.0)
true_model = LinearGaussianStateSpaceModel(θ)
_, x, y = sample(rng, true_model, 200);

# Here are the latent and obseravation timeseries
plot(x; label="x", xlabel="t")
plot!(y; seriestype=:scatter, label="y", xlabel="t", mc=:red, ms=2, ma=0.5)

# `AdvancedPS` subscribes to the `AbstractMCMC` API. To sample we just need to define a Particle Gibbs kernel
# and a model interface. 
pgas = AdvancedPS.PGAS(20)
chains = sample(rng, true_model(y), pgas, 500; progress=false);
#md nothing #hide

# 
particles = hcat([chain.trajectory.model.X for chain in chains]...)
mean_trajectory = mean(particles; dims=2);
#md nothing #hide

# This toy model is small enough to inspect all the generated traces:
scatter(particles; label=false, opacity=0.01, color=:black, xlabel="t", ylabel="state")
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)

# We used a particle gibbs kernel with the ancestor updating step which should help with the particle 
# degeneracy problem and improve the mixing. 
# We can compute the update rate of $x_t$ vs $t$ defined as the proportion of times $t$ where $x_t$ gets updated:
update_rate = sum(abs.(diff(particles; dims=2)) .> 0; dims=2) / length(chains)
#md nothing #hide

# and compare it to the theoretical value of $1 - 1/Nₚ$. 
plot(
    update_rate;
    label=false,
    ylim=[0, 1],
    legend=:bottomleft,
    xlabel="Iteration",
    ylabel="Update rate",
)
hline!([1 - 1 / length(chains)]; label="N: $(length(chains))")
