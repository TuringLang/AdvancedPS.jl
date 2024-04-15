# # Gaussian Process State-Space Model (GP-SSM)
using LinearAlgebra
using Random
using AdvancedPS
using AbstractGPs
using Plots
using Distributions
using Libtask
using SSMProblems

Parameters = @NamedTuple begin
    a::Float64
    q::Float64
    kernel
end

mutable struct GPSSM <: SSMProblems.AbstractStateSpaceModel
    X::Vector{Float64}
    observations::Vector{Float64}
    θ::Parameters

    GPSSM(params::Parameters) = new(Vector{Float64}(), params)
    GPSSM(y::Vector{Float64}, params::Parameters) = new(Vector{Float64}(), y, params)
end

seed = 1
T = 100
Nₚ = 20
Nₛ = 250
a = 0.9
q = 0.5

params = Parameters((a, q, SqExponentialKernel()))

f(θ::Parameters, x, t) = Normal(θ.a * x, θ.q)
h(θ::Parameters) = Normal(0, θ.q)
g(θ::Parameters, x, t) = Normal(0, exp(0.5 * x)^2)

rng = Random.MersenneTwister(seed)

x = zeros(T)
y = similar(x)
x[1] = rand(rng, h(params))
for t in 1:T
    if t < T
        x[t + 1] = rand(rng, f(params, x[t], t))
    end
    y[t] = rand(rng, g(params, x[t], t))
end

function gp_update(model::GPSSM, state, step)
    gp = GP(model.θ.kernel)
    prior = gp(1:(step - 1))
    post = posterior(prior, model.X[1:(step - 1)])
    μ, σ = mean_and_cov(post, [step])
    return Normal(μ[1], σ[1])
end

SSMProblems.transition!!(rng::AbstractRNG, model::GPSSM) = rand(rng, h(model.θ))
function SSMProblems.transition!!(rng::AbstractRNG, model::GPSSM, state, step)
    return rand(rng, gp_update(model, state, step))
end

function SSMProblems.emission_logdensity(model::GPSSM, state, step)
    return logpdf(g(model.θ, state, step), model.observations[step])
end
function SSMProblems.transition_logdensity(model::GPSSM, prev_state, current_state, step)
    return logpdf(gp_update(model, prev_state, step), current_state)
end

AdvancedPS.isdone(::GPSSM, step) = step > T

model = GPSSM(y, params)
pg = AdvancedPS.PGAS(Nₚ)
chains = sample(rng, model, pg, Nₛ)

particles = hcat([chain.trajectory.model.X for chain in chains]...)
mean_trajectory = mean(particles; dims=2);

scatter(particles; label=false, opacity=0.01, color=:black, xlabel="t", ylabel="state")
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)
