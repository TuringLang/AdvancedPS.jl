# # Levy-SSM latent state inference
using AdvancedPS: SSMProblems
using AdvancedPS
using Random
using Plots
using Distributions
using AdvancedPS
using LinearAlgebra
using SSMProblems

struct GammaProcess
    C::Float64
    β::Float64
    tol::Float64
end

struct GammaPath{T}
    jumps::Vector{T}
    times::Vector{T}
end

struct LangevinDynamics{T}
    A::Matrix{T}
    L::Vector{T}
    θ::T
    H::Vector{T}
    σe::T
end

struct NormalMeanVariance{T}
    μ::T
    σ::T
end

function simulate(
    rng::AbstractRNG,
    process::GammaProcess,
    rate::Float64,
    start::Float64,
    finish::Float64,
    t0::Float64=0.0,
)
    let β = process.β, C = process.C, tolerance = process.tol
        jumps = Float64[]
        last_jump = Inf
        t = t0
        truncated = last_jump < tolerance
        while !truncated
            t += rand(rng, Exponential(1.0 / rate))
            xi = 1.0 / (β * (exp(t / C) - 1))
            prob = (1.0 + β * xi) * exp(-β * xi)
            if rand(rng) < prob
                push!(jumps, xi)
                last_jump = xi
            end
            truncated = last_jump < tolerance
        end
        times = rand(rng, Uniform(start, finish), length(jumps))
        return GammaPath(jumps, times)
    end
end

function integral(times::Array{Float64}, path::GammaPath)
    let jumps = path.jumps, jump_times = path.times
        return [sum(jumps[jump_times .<= t]) for t in times]
    end
end

# Gamma Process
C = 1.0
β = 1.0
ϵ = 1e-10
process = GammaProcess(C, β, ϵ)

# Normal Mean-Variance representation
μw = 0.0
σw = 1.0
nvm = NormalMeanVariance(μw, σw)

# Levy SSM with Langevin dynamics
#   dx(t) = A x(t) dt + L dW(t)
#   y(t)  = H x(t) + ϵ(t)
θ = -0.5
A = [
    0.0 1.0
    0.0 θ
]
L = [0.0; 1.0]
σe = 1.0
H = [1.0, 0]
dyn = LangevinDynamics(A, L, θ, H, σe)

# Simulation parameters
start, finish = 0, 100
N = 200
ts = range(start, finish; length=N)
seed = 4
rng = Random.MersenneTwister(seed)
Np = 10
Ns = 10

f(dt, θ) = exp(θ * dt)
function Base.exp(dyn::LangevinDynamics, dt::Real)
    let θ = dyn.θ
        f_val = f(dt, θ)
        return [1.0 (f_val - 1)/θ; 0 f_val]
    end
end

function meancov(
    t::T, dyn::LangevinDynamics, path::GammaPath, nvm::NormalMeanVariance
) where {T<:Real}
    μ = zeros(T, 2)
    Σ = zeros(T, (2, 2))
    let times = path.times, jumps = path.jumps, μw = nvm.μ, σw = nvm.σ
        for (v, z) in zip(times, jumps)
            ft = exp(dyn, (t - v)) * dyn.L
            μ += ft .* μw .* z
            Σ += ft * transpose(ft) .* σw^2 .* z
        end
        return μ, Σ
    end
end

X = zeros(Float64, (N, 2))
Y = zeros(Float64, N)
for (i, t) in enumerate(ts)
    if i > 1
        s = ts[i - 1]
        dt = t - s
        path = simulate(rng, process, dt, s, t, ϵ)
        μ, Σ = meancov(t, dyn, path, nvm)
        X[i, :] .= rand(rng, MultivariateNormal(exp(dyn, dt) * X[i - 1, :] + μ, Σ))
    end

    let H = dyn.H, σe = dyn.σe
        Y[i] = transpose(H) * X[i, :] + rand(rng, Normal(0, σe))
    end
end

# AdvancedPS
Parameters = @NamedTuple begin
    dyn::LangevinDynamics
    process::GammaProcess
    nvm::NormalMeanVariance
    times::Vector{Float64}
end

struct MixedState{T}
    x::Vector{T}
    path::GammaPath{T}
end

mutable struct LevyLangevin <: SSMProblems.AbstractStateSpaceModel
    X::Vector{MixedState{Float64}}
    observations::Vector{Float64}
    θ::Parameters
    LevyLangevin(θ::Parameters) = new(Vector{MixedState{Float64}}(), θ)
    function LevyLangevin(y::Vector{Float64}, θ::Parameters)
        return new(Vector{MixedState{Float64}}(), y, θ)
    end
end

function SSMProblems.transition!!(rng::AbstractRNG, model::LevyLangevin)
    return MixedState(
        rand(rng, MultivariateNormal([0, 0], I)), GammaPath(Float64[], Float64[])
    )
end

function SSMProblems.transition!!(
    rng::AbstractRNG, model::LevyLangevin, state::MixedState, step
)
    times = model.θ.times
    s = times[step - 1]
    t = times[step]
    dt = t - s
    path = simulate(rng, model.θ.process, dt, s, t)
    μ, Σ = meancov(t, model.θ.dyn, path, model.θ.nvm)
    Σ += 1e-6 * I
    return MixedState(rand(rng, MultivariateNormal(exp(dyn, dt) * state.x + μ, Σ)), path)
end

function SSMProblems.transition_logdensity(
    model::LevyLangevin, prev_state::MixedState, current_state::MixedState, step
)
    times = model.θ.times
    s = times[step - 1]
    t = times[step]
    dt = t - s
    path = simulate(rng, model.θ.process, dt, s, t)
    μ, Σ = meancov(t, model.θ.dyn, path, model.θ.nvm)
    Σ += 1e-6 * I
    return logpdf(MultivariateNormal(exp(dyn, dt) * prev_state.x + μ, Σ), current_state.x)
end

function SSMProblems.emission_logdensity(model::LevyLangevin, state::MixedState, step)
    return logpdf(Normal(transpose(H) * state.x, σe), model.observations[step])
end

AdvancedPS.isdone(model::LevyLangevin, step) = step > length(model.θ.times)

θ₀ = Parameters((dyn, process, nvm, ts))
model = LevyLangevin(Y, θ₀)
pg = AdvancedPS.PGAS(Np)
chains = sample(rng, model, pg, Ns; progress=false);

# Concat all sampled states
particles = hcat([chain.trajectory.model.X for chain in chains]...)
marginal_states = map(s -> s.x, particles);
jump_times = map(s -> s.path.times, particles);
jump_intensities = map(s -> s.path.jumps, particles);

# Plot marginal state and jump intensities for one trajectory
p1 = plot(
    ts,
    [state[1] for state in marginal_states[:, end]];
    color=:darkorange,
    label="Marginal State (x1)",
)
plot!(
    ts,
    [state[2] for state in marginal_states[:, end]];
    color=:dodgerblue,
    label="Marginal State (x2)",
)

p2 = scatter(
    vcat([t for t in jump_times[:, end]]...),
    vcat([j for j in jump_intensities[:, end]]...);
    color=:darkorange,
    label="Jumps",
)

plot(
    p1, p2; plot_title="Marginal State and Jump Intensities", layout=(2, 1), size=(600, 600)
)

# Plot mean trajectory with standard deviation
mean_trajectory = transpose(hcat(mean(marginal_states; dims=2)...))
std_trajectory = dropdims(std(stack(marginal_states); dims=3); dims=3)

ps = []
for d in 1:2
    p = plot(
        mean_trajectory[:, d];
        ribbon=2 * std_trajectory[:, d]',
        color=:darkorange,
        label="Mean Trajectory (±2σ)",
        fillalpha=0.2,
        title="Marginal State Trajectories (X$d)",
    )
    plot!(p, X[:, d]; color=:dodgerblue, label="True Trajectory")
    push!(ps, p)
end
plot(ps...; layout=(2, 1), size=(600, 600))
