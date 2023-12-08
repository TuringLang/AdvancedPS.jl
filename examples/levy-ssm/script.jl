# # The Levy State Space Model (Godsill et al. 2020)
using AbstractMCMC
using AdvancedPS
using Distributions
using Random
using Plots

include("simulation.jl")

Parameters = @NamedTuple begin
    μ_W::Float64  # Subordinator process mean
    σ_W::Float64  # Subordinator process variance
    β::Float64    # Tempering parameter
    C::Float64    # Jump density
    A::Float64    # Drift term
    h::Float64    # Diffusion term
    X0::Float64   # Initial value
    σ_n::Float64  # Observation noise
end

# Simulation parameters
T = 100    # Time horizon
res = 100  # Sampling resolution
seed = 1   # Random seed
rng = Random.MersenneTwister(seed)

# Model parameters
μ_W = 0.0
σ_W = 1.0
β = 1.0
C = 1.0
A = -1.0
h = 1.0
X0 = 0.0
σ_n = 0.5
θ = Parameters((μ_W, σ_W, β, C, A, h, X0, σ_n))

# Sampling parameters
N_p = 10  # Number of particles
N_s = 100  # Number of samples

# Simulate data
# TODO: pass in RNG
ngp = NormalGammaProcess(μ_W, σ_W, β, C, T)
ts = range(0, T, length=res)
SDE = UnivariateFiniteSDE(A, h, T, X0, ngp)
x, __... = simulate(SDE, res)
y = x + σ_n * randn(rng, res)

# Plot the data
plot(ts, x; label="Latent State", xlabel="t", linewidth=2)
# Scatter y
scatter!(ts, y; label="Observation")

# Define state space model
mutable struct LevySSM <: AdvancedPS.AbstractStateSpaceModel
    X::Vector{Float64}
    θ::Parameters
    LevySSM(θ::Parameters) = new(Float64[], θ)
end

# The dynamics of the model is defined through the `AbstractStateSpaceModel` interface:
AdvancedPS.initialization(model::LevySSM) = Dirac(model.θ.X0)
# TODO: good God this is ugly but it works for now
function AdvancedPS.transition(model::LevySSM, state, step)
    # TODO: remove global scope
    one_step_SDE = UnivariateFiniteSDE(model.θ.A, model.θ.h, T / res, state, ngp)
    return Dirac(simulate(one_step_SDE, 2)[1][2])
end
function AdvancedPS.observation(model::LevySSM, state, step)
    return logpdf(Normal(state, model.θ.σ_n), y[step])  # TODO: avoid global variable
end
AdvancedPS.isdone(::LevySSM, step) = step > T

# Run particle Gibbs
model = LevySSM(θ)
pg = AdvancedPS.PG(N_p, 1.0)
chains = sample(rng, model, pg, N_s; progress=true);

particles = hcat([chain.trajectory.model.X for chain in chains]...)
mean_trajectory = mean(particles; dims=2);

# Plot mean trajectories
scatter(
    particles; label=false, opacity=0.5, color=:black, xlabel="t", ylabel="state"
)
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)
