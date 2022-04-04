# # Gaussian State Space Model
using AdvancedPS
using Random
using Distributions
using Plots

# # Model definition
# 
# We consider the following linear state-space model with Gaussian innovations. The model is one-dimensional and is given
# by the following transition equations.
# 
# ```
#  x_{t+1} = a x_{t} + \epsilon_t \quad \epsilon_t \sim \mathcal{N}(0,q^2)
#  y_{t} = x_{t} + \nu_{t} \quad \nu \sim \mathcal{N}(0, r^2) 
# ```
# We're interested in sampling from the latent state `x_t`. We assume the static paramters `\theta = (a, q^2, r^2)` are known.
# 

# Parameters
# We define a small 
Parameters = @NamedTuple begin
    a::Float64
    q::Float64
    r::Float64
end

a = 0.9   # Scale
q = 0.32  # State variance
r = 1     # Observation variance
Tₘ = 300  # Number of observation
Nₚ = 50   # Number of particles
Nₛ = 1000 # Number of samples
seed = 9  # Reproduce everything

θ₀ = Parameters((a, q, r))

mutable struct NonLinearTimeSeries <: AdvancedPS.AbstractStateSpaceModel
    X::Vector{Float64}
    θ::Parameters
    NonLinearTimeSeries(θ::Parameters) = new(Vector{Float64}(), θ)
    NonLinearTimeSeries() = new(Vector{Float64}(), θ₀)
end

f(m::NonLinearTimeSeries, state, t) = Normal(m.θ.a * state, m.θ.q) # Transition density
g(m::NonLinearTimeSeries, state, t) = Normal(state, m.θ.r) # Observation density
f₀(m::NonLinearTimeSeries) = Normal(0, m.θ.q / (1 - m.θ.a^2)) # Initial state density

AdvancedPS.initialization(model::NonLinearTimeSeries) = f₀(model)
AdvancedPS.transition(model::NonLinearTimeSeries, state, step) = f(model, state, step)
function AdvancedPS.observation(model::NonLinearTimeSeries, state, step)
    return logpdf(g(model, state, step), y[step])
end # Return log-pdf of the obs
AdvancedPS.isdone(::NonLinearTimeSeries, step) = step > Tₘ

# Generate some synthetic data
rng = Random.MersenneTwister(seed)

x = zeros(Tₘ)
y = zeros(Tₘ)

reference = NonLinearTimeSeries(θ₀) # Reference model
x[1] = rand(rng, f₀(reference))
for t in 1:Tₘ
    if t < Tₘ
        x[t + 1] = rand(rng, f(reference, x[t], t))
    end
    y[t] = rand(rng, g(reference, x[t], t))
end

plot(x; label="x", color=:black)
plot(y; label="y", color=:black)

# Setup up a particle Gibbs sampler
model = NonLinearTimeSeries(θ₀)
pgas = AdvancedPS.PGAS(Nₚ)
chains = sample(rng, model, pgas, Nₛ)

particles = hcat([chain.trajectory.model.X for chain in chains]...) # Concat all sampled states
mean_trajectory = mean(particles; dims=2)

scatter(particles; label=false, opacity=0.01, color=:black)
plot!(x; color=:red, label="Original Trajectory")
plot!(mean_trajectory; color=:blue, label="Posterior mean", opacity=0.9)

# Compute mixing rate
update_rate = sum(abs.(diff(particles; dims=2)) .> 0; dims=2) / Nₛ
plot(update_rate; label=false, ylim=[0, 1])
hline!([1 - 1 / Nₚ]; label=false)
