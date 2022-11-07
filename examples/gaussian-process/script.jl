# # Gaussian Process State-Space Model (GP-SSM)
using LinearAlgebra
using Random
using AdvancedPS
using AbstractGPs
using Plots
using Distributions
using Libtask

Parameters = @NamedTuple begin
    a::Float64
    q::Float64
    kernel
end

mutable struct GPSSM <: AdvancedPS.AbstractStateSpaceModel
    X::Vector{Float64}
    θ::Parameters

    GPSSM(params::Parameters) = new(Vector{Float64}(), params)
end

seed = 1
T = 100
Nₚ = 20
Nₛ = 250
a = 0.9
q = 0.5

params = Parameters((a, q, SqExponentialKernel()))

f(model::GPSSM, x, t) = Normal(model.θ.a * x, model.θ.q)
h(model::GPSSM) = Normal(0, model.θ.q)
g(model::GPSSM, x, t) = Normal(0, exp(0.5 * x)^2)

rng = Random.MersenneTwister(seed)
ref_model = GPSSM(params)

x = zeros(T)
y = similar(x)
x[1] = rand(rng, h(ref_model))
for t in 1:T
    if t < T
        x[t + 1] = rand(rng, f(ref_model, x[t], t))
    end
    y[t] = rand(rng, g(ref_model, x[t], t))
end

function gp_update(model::GPSSM, state, step)
    gp = GP(model.θ.kernel)
    prior = gp(1:(step - 1))
    post = posterior(prior, model.X[1:(step - 1)])
    μ, σ = mean_and_cov(post, [step])
    return Normal(μ[1], σ[1])
end

AdvancedPS.initialization(::GPSSM) = h(model)
AdvancedPS.transition(model::GPSSM, state, step) = gp_update(model, state, step)
AdvancedPS.observation(model::GPSSM, state, step) = logpdf(g(model, state, step), y[step])
AdvancedPS.isdone(::GPSSM, step) = step > T

model = GPSSM(params)
pg = AdvancedPS.PGAS(Nₚ)
chains = sample(rng, model, pg, Nₛ)

particles = hcat([chain.trajectory.model.X for chain in chains]...)
mean_trajectory = mean(particles; dims=2);

scatter(particles; label=false, opacity=0.01, color=:black, xlabel="t", ylabel="state")
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)
