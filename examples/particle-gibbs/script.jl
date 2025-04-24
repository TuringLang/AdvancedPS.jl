# # Particle Gibbs for non-linear models
using AdvancedPS
using Random
using Distributions
using Plots
using AbstractMCMC
using Random123
using SSMProblems

"""
    plot_update_rate(update_rate, N)

Plot empirical update rate against theoretical value
"""
function plot_update_rate(update_rate::AbstractVector{Float64}, Nₚ::Int)
    plt = plot(
        update_rate;
        label=false,
        ylim=[0, 1],
        legend=:bottomleft,
        xlabel="Iteration",
        ylabel="Update rate",
    )
    return hline!(plt, [1 - 1 / Nₚ]; label="N: $(Nₚ)")
end

"""
    update_rate(trajectories, N)

Compute latent state update rate
"""
function update_rate(particles::AbstractMatrix{Float64}, Nₛ)
    return sum(abs.(diff(particles; dims=2)) .> 0; dims=2) / Nₛ
end
#md nothing #hide

# We consider the following stochastic volatility model:
# 
# ```math
#  x_{t+1} = a x_t + v_t \quad v_{t} \sim \mathcal{N}(0, q^2)
# ```
# ```math
#  y_{t} = e_t \exp(\frac{1}{2}x_t) \quad e_t \sim \mathcal{N}(0, 1) 
# ```
#
# We can reformulate the above in terms of transition and observation densities:
# ```math
#  x_{t+1} \sim f_{\theta}(x_{t+1}|x_t) = \mathcal{N}(a x_t, q^2)
# ```
# ```math
#  y_t \sim g_{\theta}(y_t|x_t) = \mathcal{N}(0, \exp(\frac{1}{2}x_t)^2)
# ```
# with the initial distribution $f_0(x) = \mathcal{N}(0, q^2)$.
# Here we assume the static parameters $\theta = (a^2, q^2)$ are known and we are only interested in sampling from the latent state $x_t$. 
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

struct StochasticVolatility{T<:Real} <: SSMProblems.ObservationProcess{T,T} end

function SSMProblems.distribution(::StochasticVolatility{T}, ::Int, state) where {T<:Real}
    return Normal(zero(T), exp((1 / 2) * state))
end

function LinearGaussianStochasticVolatilityModel(a::T, q::T) where {T<:Real}
    dyn = LinearGaussianDynamics(a, q)
    obs = StochasticVolatility{T}()
    return SSMProblems.StateSpaceModel(dyn, obs)
end
#md nothing #hide

# Let's simulate some data
rng = Random.MersenneTwister(1234)
true_model = LinearGaussianStochasticVolatilityModel(0.9, 0.5)
_, x, y = sample(rng, true_model, 200);

# Here are the latent and observation series:
plot(x; label="x", xlabel="t")

# 
plot(y; label="y", xlabel="t")

# Here we use the particle gibbs kernel without adaptive resampling.
model = true_model(y)
pg = AdvancedPS.PG(20, 1.0)
chains = sample(rng, model, pg, 200; progress=false);
#md nothing #hide

particles = hcat([chain.trajectory.model.X for chain in chains]...) # Concat all sampled states
mean_trajectory = mean(particles; dims=2);
#md nothing #hide

# We can now plot all the generated traces.
# Beyond the last few timesteps all the trajectories collapse into one. Using the ancestor updating step can help with the degeneracy problem, as we show below.
scatter(
    particles[:, 1:50]; label=false, opacity=0.5, color=:black, xlabel="t", ylabel="state"
)
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)

# We can also check the mixing as defined in the Gaussian State Space model example. As seen on the
# scatter plot above, we are mostly left with a single trajectory before timestep 150. The orange 
# bar is the optimal mixing rate for the number of particles we use.
plot_update_rate(update_rate(particles, 200)[:, 1], 20)

# Let's see if ancestor sampling can help with the degeneracy problem. We use the same number of particles, but replace the sampler with PGAS. 
pgas = AdvancedPS.PGAS(20)
chains = sample(rng, model, pgas, 200; progress=false);
particles = hcat([chain.trajectory.model.X for chain in chains]...);
mean_trajectory = mean(particles; dims=2);

# The ancestor sampling has helped with the degeneracy problem and we now have a much more diverse set of trajectories, also at earlier time periods.
scatter(
    particles[:, 1:50]; label=false, opacity=0.5, color=:black, xlabel="t", ylabel="state"
)
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)

# The update rate is now much higher throughout time.
plot_update_rate(update_rate(particles, 200)[:, 1], 20)
