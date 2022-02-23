# # Gaussian State Space Model
using AdvancedPS
using Random
using Distributions
using Plots

# Parameters
θ = @NamedTuple begin
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

θ₀ = θ((a, q, r))

f(x, t) = Normal(a * x, q) # Transition density
g(x, t) = Normal(x, r)     # Observation density
f₀ = Normal(0, q)          # Initial state density

mutable struct NonLinearTimeSeries <: AdvancedPS.AbstractStateSpaceModel
    X::Vector{Float64}
    θ::θ
    NonLinearTimeSeries(θ::θ) = new(Vector{Float64}(), θ)
    NonLinearTimeSeries() = new(Vector{Float64}(), θ₀)
end

AdvancedPS.initialization(::NonLinearTimeSeries) = f₀
AdvancedPS.transition(::NonLinearTimeSeries, state, step) = f(state, step)
AdvancedPS.observation(::NonLinearTimeSeries, state, step) = logpdf(g(state, step), y[step]) # Return log-pdf of the obs
AdvancedPS.isdone(::NonLinearTimeSeries, step) = step > Tₘ

# Generate some synthetic data
rng = Random.MersenneTwister(seed)

x = zeros(Tₘ)
y = zeros(Tₘ)

x[1] = rand(rng, f₀)
for t in 1:Tₘ
    if t < Tₘ
        x[t + 1] = rand(rng, f(x[t], t))
    end
    y[t] = rand(rng, g(x[t], t))
end

plot(x, label="x", color=:black)
plot(y, label="y", color=:black)

# Turing
model = NonLinearTimeSeries()
pgas = AdvancedPS.PGAS(Nₚ)
chains = sample(rng, model, pgas, Nₛ)

particles = hcat([v.trajectory.f.X for v in chains]...) # Concat all sampled states
mean_trajectory = mean(particles, dims=2)

scatter(particles, label=false, opacity=0.01, color=:black)
plot!(x, color=:red, label="Original Trajectory")
plot!(mean_trajectory, color=:blue, label="Posterior mean", opacity=.9)

# Compute mixing rate
update_rate = sum(abs.(diff(particles, dims=2)) .> 0, dims=2)/Nₛ
plot(update_rate, label=false, ylim=[0,1])
hline!([1-1/Nₚ], label=false)