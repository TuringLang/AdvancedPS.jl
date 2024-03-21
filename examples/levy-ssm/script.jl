# # Levy-SSM 
using Random
using Plots
using Distributions
using AdvancedPS
using LinearAlgebra

struct GammaProcess
    C::Float64
    beta::Float64
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
    process::GammaProcess,
    rate::Float64,
    start::Float64,
    finish::Float64,
    truncation::Float64,
    t0::Float64=0.0,
)
    let beta = process.beta, C = process.C
        jumps = Float64[]
        last_jump = Inf
        t = t0
        truncated = last_jump < truncation
        while !truncated
            t += rand(Exponential(1.0 / rate))
            xi = 1.0 / (beta * (exp(t / C) - 1))
            prob = (1.0 + beta * xi) * exp(-beta * xi)
            if rand() < prob
                push!(jumps, xi)
                last_jump = xi
            end
            truncated = last_jump < truncation
        end
        times = rand(Uniform(start, finish), length(jumps))
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
beta = 1.0
ϵ = 1e-10
process = GammaProcess(C, beta)

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
        path = simulate(process, dt, s, t, ϵ)
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

mutable struct LevyLangevin <: AdvancedPS.AbstractStateSpaceModel
    X::Vector{MixedState{Float64}}
    θ::Parameters
    LevyLangevin(θ::Parameters) = new(Vector{MixedState{Float64}}(), θ)
end

struct InitialDistribution end
function Base.rand(rng::Random.AbstractRNG, ::InitialDistribution)
    return MixedState(rand(MultivariateNormal([0, 0], I)), GammaPath(Float64[], Float64[]))
end

struct TransitionDistribution{T}
    current_state::MixedState{T}
    model::LevyLangevin
    s::T
    t::T
end

function Base.rand(rng::Random.AbstractRNG, td::TransitionDistribution)
    let model = td.model, s = td.s, t = td.t, state = td.current_state
        dt = t - s
        path = simulate(model.θ.process, dt, s, t, ϵ)
        μ, Σ = meancov(t, model.θ.dyn, path, model.θ.nvm)
        Σ += 1e-6 * I
        return MixedState(
            rand(rng, MultivariateNormal(exp(dyn, dt) * state.x + μ, Σ)), path
        )
    end
end

# Required for ancestor sampling in PGAS
function Distributions.logpdf(td::TransitionDistribution, state::MixedState)
    let model = td.model, s = td.s, t = td.t, state = td.current_state
        dt = t - s
        path = simulate(model.θ.process, dt, s, t, ϵ)
        μ, Σ = meancov(t, model.θ.dyn, path, model.θ.nvm)
        Σ += 1e-6 * I
        return logpdf(MultivariateNormal(exp(dyn, dt) * state.x + μ, Σ), state.x)
    end
end

θ₀ = Parameters((dyn, process, nvm, ts))

function AdvancedPS.initialization(::LevyLangevin)
    return InitialDistribution()
end
function AdvancedPS.transition(model::LevyLangevin, state::MixedState, step)
    times = model.θ.times
    s = times[step - 1]
    t = times[step]
    return TransitionDistribution(state, model, s, t)
end

function AdvancedPS.observation(model::LevyLangevin, state::MixedState, step)
    return logpdf(Normal(transpose(H) * state.x, σe), Y[step])
end
AdvancedPS.isdone(::LevyLangevin, step) = step > length(ts)

model = LevyLangevin(θ₀)
pg = AdvancedPS.PG(Np)
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

plot(
    mean_trajectory;
    ribbon=std_trajectory',
    color=:darkorange,
    label="Mean trajectory",
    opacity=0.3,
    title="Inference Quality",
)
plot!(
    mean_trajectory; color=:dodgerblue, label="Original Trajectory", title="Path Degeneracy"
)
