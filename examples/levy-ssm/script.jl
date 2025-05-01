# # Levy-SSM latent state inference
using Random
using Plots
using Distributions
using AdvancedPS
using LinearAlgebra
using SSMProblems

struct GammaProcess{T}
    C::T
    β::T
    tol::T
    GammaProcess(C::T, β::T; ϵ::T=1e-10) where {T<:Real} = new{T}(C, β, ϵ)
end

struct GammaPath{T}
    jumps::Vector{T}
    times::Vector{T}
end

function simulate(
    rng::AbstractRNG, process::GammaProcess{T}, rate::T, start::T, finish::T, t0::T=zero(T)
) where {T<:Real}
    let β = process.β, C = process.C, tolerance = process.tol
        jumps = T[]
        last_jump = Inf
        t = t0
        truncated = last_jump < tolerance
        while !truncated
            t += rand(rng, Exponential(one(T) / rate))
            xi = one(T) / (β * (exp(t / C) - one(T)))
            prob = (one(T) + β * xi) * exp(-β * xi)
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

function integral(times::Array{<:Real}, path::GammaPath)
    let jumps = path.jumps, jump_times = path.times
        return [sum(jumps[jump_times .<= t]) for t in times]
    end
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

f(dt, θ) = exp(θ * dt)
function Base.exp(dyn::LangevinDynamics{T}, dt::T) where {T<:Real}
    let θ = dyn.θ
        f_val = f(dt, θ)
        return [one(T) (f_val - 1)/θ; zero(T) f_val]
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

        # Guarantees positive semi-definiteness
        return μ, Σ + T(1e-6) * I
    end
end

struct LevyLangevin{T} <: LatentDynamics{T,Vector{T}}
    dt::T
    dyn::LangevinDynamics{T}
    process::GammaProcess{T}
    nvm::NormalMeanVariance{T}
end

function SSMProblems.distribution(proc::LevyLangevin{T}) where {T<:Real}
    return MultivariateNormal(zeros(T, 2), I)
end

function SSMProblems.distribution(proc::LevyLangevin{T}, step::Int, state) where {T<:Real}
    dt = proc.dt
    path = simulate(rng, proc.process, dt, (step - 1) * dt, step * dt)
    μ, Σ = meancov(step * dt, proc.dyn, path, proc.nvm)
    return MultivariateNormal(exp(proc.dyn, dt) * state + μ, Σ)
end

struct LinearGaussianObservation{T<:Real} <: ObservationProcess{T,T}
    H::Vector{T}
    R::T
end

function SSMProblems.distribution(proc::LinearGaussianObservation, step::Int, state)
    return Normal(transpose(proc.H) * state, proc.R)
end

function LevyModel(dt, θ, σe, C, β, μw, σw; ϵ=1e-10)
    A = [0.0 1.0; 0.0 θ]
    L = [0.0; 1.0]
    H = [1.0, 0]

    dyn = LevyLangevin(
        dt,
        LangevinDynamics(A, L, θ, H, σe),
        GammaProcess(C, β; ϵ),
        NormalMeanVariance(μw, σw),
    )

    obs = LinearGaussianObservation(H, σe)
    return StateSpaceModel(dyn, obs)
end

# Levy SSM with Langevin dynamics
# ```math
#   dx_{t} = A x_{t} dt + L dW_{t}
# ```
# ```math
#   y_{t} = H x_{t} + ϵ{t}
# ```

# Simulation parameters
N = 200
ts = range(0, 100; length=N)
levyssm = LevyModel(step(ts), θ, 1.0, 1.0, 1.0, 0.0, 1.0);

# Simulate data
rng = Random.MersenneTwister(1234);
_, X, Y = sample(rng, levyssm, N);

# Run sampler
pg = AdvancedPS.PGAS(50);
chains = sample(rng, levyssm(Y), pg, 100);

# Concat all sampled states
marginal_states = hcat([chain.trajectory.model.X for chain in chains]...)

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

# TODO: collect jumps from the model
p2 = scatter([], []; color=:darkorange, label="Jumps")

plot(
    p1, p2; plot_title="Marginal State and Jump Intensities", layout=(2, 1), size=(600, 600)
)

# Plot mean trajectory with standard deviation
mean_trajectory = transpose(hcat(mean(marginal_states; dims=2)...))
std_trajectory = dropdims(std(stack(marginal_states); dims=3); dims=3)

ps = []
for d in 1:2
    p = plot(
        ts,
        mean_trajectory[:, d];
        ribbon=2 * std_trajectory[:, d]',
        color=:darkorange,
        label="Mean Trajectory (±2σ)",
        fillalpha=0.2,
        title="Marginal State Trajectories (X$d)",
    )
    plot!(p, ts, getindex.(X, d); color=:dodgerblue, label="True Trajectory")
    push!(ps, p)
end
plot(ps...; layout=(2, 1), size=(600, 600))
