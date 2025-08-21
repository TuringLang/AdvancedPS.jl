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
            t += rand(rng, Exponential(1 / rate))
            xi = 1 / (β * (exp(t / C) - 1))
            prob = (1 + β * xi) * exp(-β * xi)
            if rand(rng) < prob
                push!(jumps, xi)
                last_jump = xi
            end
            truncated = last_jump < tolerance
        end
        return GammaPath(jumps, rand(rng, Uniform(start, finish), length(jumps)))
    end
end

function integral(times::Array{<:Real}, path::GammaPath)
    let jumps = path.jumps, jump_times = path.times
        return [sum(jumps[jump_times .<= t]) for t in times]
    end
end

struct LangevinDynamics{AT<:AbstractMatrix,LT<:AbstractVector,θT<:Real}
    A::AT
    L::LT
    θ::θT
end

function Base.exp(dyn::LangevinDynamics, dt)
    f_val = exp(dyn.θ * dt)
    return [1 (f_val - 1)/dyn.θ; 0 f_val]
end

function meancov(t, dyn::LangevinDynamics, path::GammaPath, dist::Normal)
    fts = exp.(Ref(dyn), (t .- path.times)) .* Ref(dyn.L)
    μ = sum(@. fts * mean(dist) * path.jumps)
    Σ = sum(@. fts * transpose(fts) * var(dist) * path.jumps)

    # Guarantees positive semi-definiteness
    return μ, Σ + eltype(Σ)(1e-6) * I
end

struct LevyPrior{XT<:AbstractVector,ΣT<:AbstractMatrix} <: StatePrior
    μ::XT
    Σ::ΣT
end

SSMProblems.distribution(proc::LevyPrior) = MvNormal(proc.μ, proc.Σ)

struct LevyLangevin{T<:Real,LT<:LangevinDynamics,ΓT<:GammaProcess,DT<:Normal} <:
       SSMProblems.LatentDynamics
    dt::T
    dyn::LT
    process::ΓT
    dist::DT
end

function SSMProblems.distribution(proc::LevyLangevin, step::Int, state)
    dt = proc.dt
    path = simulate(rng, proc.process, dt, (step - 1) * dt, step * dt)
    μ, Σ = meancov(step * dt, proc.dyn, path, proc.dist)
    return MvNormal(exp(proc.dyn, dt) * state + μ, Σ)
end

struct LinearGaussianObservation{HT<:AbstractVector,RT<:Real} <: SSMProblems.ObservationProcess
    H::HT
    R::RT
end

function SSMProblems.distribution(proc::LinearGaussianObservation, ::Int, state)
    return Normal(transpose(proc.H) * state, proc.R)
end

function LevyModel(dt, θ, σe, C, β, μw, σw; kwargs...)
    dyn = LevyLangevin(
        dt,
        LangevinDynamics([0 1; 0 θ], [0; 1], θ),
        GammaProcess(C, β; kwargs...),
        Normal(μw, σw),
    )

    obs = LinearGaussianObservation([1; 0], σe)
    return SSMProblems.StateSpaceModel(LevyPrior(zeros(Bool, 2), I(2)), dyn, obs)
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
levyssm = LevyModel(step(ts), -0.5, 1, 1.0, 1.0, 0, 1);

# Simulate data
rng = Random.MersenneTwister(1234);
_, X, Y = sample(rng, levyssm, N);

# Run sampler
pg = AdvancedPS.PGAS(50);
chains = sample(rng, AdvancedPS.TracedSSM(levyssm, Y), pg, 100; progress=false);

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
