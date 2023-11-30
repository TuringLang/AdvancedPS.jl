# # Code for simulating Levy processes
using Distributions
using Plots

abstract type LevyPointProcess end

# TODO: check naming conventions
function integrate(::LevyPointProcess, ts, jumps, jump_times)
    # TODO: are these already sorted
    samples = []
    for t in ts
        subset = jumps[jump_times .<= t]
        # Sum of empty array has no clear type
        if length(subset) > 0
            push!(samples, sum(subset))
        else
            push!(samples, 0.0)
        end
    end
    return samples
end

struct NormalGammaProcess <: LevyPointProcess
    μ_W::Float64  # Subordinator process mean
    σ_W::Float64  # Subordinator process variance
    β::Float64    # Tempering parameter
    C::Float64    # Jump density
    T::Float64    # Time horizon
end

function simulate(ngp::NormalGammaProcess, res::Int)
    ts = collect(range(0, ngp.T, length=res))
    # TODO: check naming against function internals
    gamma_paths, subordinator_jumps, jump_times = simulate_gamma_process(ngp, ts)

    NVM_jumps = (
        ngp.μ_W * subordinator_jumps +
        ngp.σ_W * sqrt.(subordinator_jumps) .* randn(length(subordinator_jumps))
    )

    samples = integrate(ngp, ts, NVM_jumps, jump_times)
    return samples, NVM_jumps, jump_times, subordinator_jumps
end

function simulate_gamma_process(ngp::NormalGammaProcess, ts::Vector{Float64})
    # Generate Poisson epochs over [0, T]
    # TODO: benchmark this vs. uniform sampling conditioned on N ~ Pois(T)
    poisson_epochs = []
    t = 0  # current time
    expired = false
    while !expired
        t += rand(Exponential(1 / ngp.T))
        if t < ngp.T
            push!(poisson_epochs, t)
        else
            expired = true
        end
    end

    # Generate samples
    Xs = []
    for t in poisson_epochs
        X = 1 / (ngp.β * (exp(t / ngp.C) - 1))
        p = (1 + ngp.β * X) * exp(-ngp.β * X)
        if rand() < p
            push!(Xs, X)
        end
    end

    # Generate jump times
    jump_times = []
    for __ in eachindex(Xs)
        push!(jump_times, rand() * T)
    end

    gamma_paths = integrate(ngp, ts, Xs, jump_times)
    return gamma_paths, Xs, jump_times
end

struct UnivariateFiniteSDE <: LevyPointProcess
    A::Float64   # Drift
    h::Float64   # Diffusion
    T::Float64   # Time horizon
    X0::Float64  # Initial state
    ngp::NormalGammaProcess
end

# TODO: ugly code — tidy up and make more efficient
function simulate(uf::UnivariateFiniteSDE, res::Int)
    ts = collect(range(0, uf.T, length=res))
    NVM_paths, NVM_jumps, jump_times, subordinator_jumps = simulate(uf.ngp, res)
    samples = []
    for t in ts
        system_jumps = NVM_jumps .* exp.(uf.A * (t .- jump_times) * uf.h)
        sample = integrate(uf, [t], system_jumps, jump_times)[1]
        push!(samples, sample)
    end
    return samples, NVM_jumps, jump_times, subordinator_jumps
end

# Test
μ_W = 0.0
σ_W = 1.0
β = 0.1
T = 10.0
C = 0.1
N = 100

ngp = NormalGammaProcess(μ_W, σ_W, β, C, T)
samples, NVM_jumps, jump_times, subordinator_jumps = simulate(ngp, N)
ts = range(0, T, length=N)
plot(ts, samples)

A = -1.0
h = 1.0
X0 = 0.0

SDE = UnivariateFiniteSDE(A, h, T, X0, ngp)
samples, NVM_jumps, jump_times, subordinator_jumps = simulate(SDE, N)
plot(ts, samples)

