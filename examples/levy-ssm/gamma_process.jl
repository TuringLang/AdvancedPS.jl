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
    # Simulate the jumps
    let beta = process.beta, C = process.C
        jumps = Float64[]
        last_jump = Inf
        t = t0
        truncated = last_jump < truncation
        while !truncated
            t += rand(Exponential(1.0 / rate))
            xi = 1.0 / (beta * (exp(t) / C - 1))
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
seed = 1
rng = Random.MersenneTwister(seed)
Np = 10
Ns = 10

f(dt, θ) = exp(θ * dt)
function Base.exp(dyn::LangevinDynamics, dt::Real)
    let θ = dyn.θ
        return [1.0 (f(dt, θ) - 1)/θ; 0 f(dt, θ)]
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
        X[i, :] = rand(MultivariateNormal(exp(dyn, dt) * X[i - 1, :] + μ, Σ))
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
    times::Vector{Float64} # Ugly, but avoids global de-ref
end

mutable struct LevyLangevin <: AdvancedPS.AbstractStateSpaceModel
    X::Vector{Vector{Float64}}
    θ::Parameters
    LevyLangevin(θ::Parameters) = new(Vector{Array{2,Float64}}(), θ)
end

θ₀ = Parameters((dyn, process, nvm, ts))

AdvancedPS.initialization(model::LevyLangevin) = MultivariateNormal([0, 0], I)
function AdvancedPS.transition(model::LevyLangevin, state, step)
    times = model.θ.times
    s = times[step - 1]
    t = times[step]
    dt = t - s
    path = simulate(model.θ.process, dt, s, t, ϵ)
    μ, Σ = meancov(t, model.θ.dyn, path, model.θ.nvm)
    return MultivariateNormal(exp(dyn, dt) * state + μ, Σ)
end

function AdvancedPS.observation(model::LevyLangevin, state, step)
    return logpdf(Normal(transpose(H) * X[step, :], σe), Y[step])
end
AdvancedPS.isdone(::LevyLangevin, step) = step > length(ts)

model = LevyLangevin(θ₀)
pg = AdvancedPS.PG(Np, 1.0)
chains = sample(rng, model, pg, Ns; progress=true);

particles = hcat([chain.trajectory.model.X for chain in chains]...) # Concat all sampled states
mean_trajectory = transpose(hcat(mean(particles; dims=2)...))

plot(X; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)
