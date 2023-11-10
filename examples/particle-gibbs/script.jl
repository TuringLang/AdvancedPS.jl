# # Particle Gibbs for non-linear models
using AdvancedPS
using SSMProblems
using Random
using Distributions
using Plots
using AbstractMCMC
using Random123

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
#  x_{t+1} = a x_t + v_t \quad v_{t} \sim \mathcal{N}(0, r^2)
# ```
# ```math
#  y_{t} = e_t \exp(\frac{1}{2}x_t) \quad e_t \sim \mathcal{N}(0, 1) 
# ```
#
# We can reformulate the above in terms of transition and observation densities:
# ```math
#  x_{t+1} \sim f_{\theta}(x_{t+1}|x_t) = \mathcal{N}(a x_t, r^2)
# ```
# ```math
#  y_t \sim g_{\theta}(y_t|x_t) = \mathcal{N}(0, \exp(\frac{1}{2}x_t)^2)
# ```
# with the initial distribution $f_0(x) = \mathcal{N}(0, q^2)$.
# Here we assume the static parameters $\theta = (q^2, r^2)$ are known and we are only interested in sampling from the latent state $x_t$. 

Base.@kwdef struct StochasticVolatilityModel <: AbstractStateSpaceModel
    a::Float64
    q::Float64
end

f₀(model::StochasticVolatilityModel) = Normal(0, model.q)
f(x::Float64, model::StochasticVolatilityModel) = Normal(model.a * x, model.q)
g(x::Float64, ::StochasticVolatilityModel) = Normal(0, exp(0.5 * x))
#md nothing #hide

# Let's simulate some data
T = 150   # Number of time steps
a = 0.9   # State Variance
q = 0.5   # Observation variance
Tₘ = 200  # Number of observation
Nₚ = 20   # Number of particles
Nₛ = 200  # Number of samples
seed = 1  # Reproduce everything

rng = Random.MersenneTwister(seed)
model = StochasticVolatilityModel(a, q)

x = zeros(Tₘ)
y = zeros(Tₘ)
x[1] = 0
for t in 1:Tₘ
    if t < Tₘ
        x[t + 1] = rand(rng, f(x[t], model))
    end
    y[t] = rand(rng, g(x[t], model))
end

# Here are the latent and observation series:
plot(x; label="x", xlabel="t")

# 
plot(y; label="y", xlabel="t")

# Define the model dynamics
function SSMProblems.transition!!(rng::AbstractRNG, model::StochasticVolatilityModel)
    return rand(rng, f₀(model))
end

function SSMProblems.transition!!(rng::AbstractRNG, model::StochasticVolatilityModel, state::Float64, ::Int)
    return rand(rng, f(state, model))
end

function SSMProblems.emission_logdensity(model::StochasticVolatilityModel, state::Float64, observation::Float64, ::Int)
    return logpdf(g(state, model), observation)
end

pg = AdvancedPS.PG(Nₚ, 1.0)
chains = sample(rng, model, pg, Nₛ; observations=y, progress=false);
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
plot_update_rate(update_rate(particles, Nₛ)[:, 1], Nₚ)

# Let's see if ancestor sampling can help with the degeneracy problem. We use the same number of particles, but replace the sampler with PGAS. 
pgas = AdvancedPS.PGAS(Nₚ)
chains = sample(rng, model, pgas, Nₛ; progress=false);
particles = hcat([chain.trajectory.model.X for chain in chains]...);
mean_trajectory = mean(particles; dims=2);

# The ancestor sampling has helped with the degeneracy problem and we now have a much more diverse set of trajectories, also at earlier time periods.
scatter(
    particles[:, 1:50]; label=false, opacity=0.5, color=:black, xlabel="t", ylabel="state"
)
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)

# The update rate is now much higher throughout time.
plot_update_rate(update_rate(particles, Nₛ)[:, 1], Nₚ)
