# # Particle Gibbs with Ancestor Sampling
using AdvancedPS
using Random
using Distributions
using Plots
using AbstractMCMC
using Random123
using Libtask: TArray
using Libtask

Parameters = @NamedTuple begin
    a::Float64
    q::Float64
    r::Float64
    T::Int
end

mutable struct NonLinearTimeSeries <: AbstractMCMC.AbstractModel
    X::TArray
    θ::Parameters
    NonLinearTimeSeries(θ::Parameters) = new(TArray(Float64, θ.T), θ)
end

f(model::NonLinearTimeSeries, state, t) = Normal(model.θ.a * state, model.θ.q) # Transition density
g(model::NonLinearTimeSeries, state, t) = Normal(state, model.θ.r)             # Observation density
f₀(model::NonLinearTimeSeries) = Normal(0, model.θ.q)    # Initial state density

# Everything is now ready to simulate some data. 

a = 0.9   # Scale
q = 0.32  # State variance
r = 1     # Observation variance
Tₘ = 300   # Number of observation
Nₚ = 20    # Number of particles
Nₛ = 500 # Number of samples
seed = 9  # Reproduce everything

θ₀ = Parameters((a, q, r, Tₘ))

rng = Random.MersenneTwister(seed)

x = zeros(Tₘ)
y = zeros(Tₘ)

reference = NonLinearTimeSeries(θ₀)
x[1] = 0
for t in 1:Tₘ
    if t < Tₘ
        x[t + 1] = rand(rng, f(reference, x[t], t))
    end
    y[t] = rand(rng, g(reference, x[t], t))
end

function (model::NonLinearTimeSeries)(rng::Random.AbstractRNG)
    x₀ = rand(rng, f₀(model))
    model.X[1] = x₀
    score = logpdf(g(model, x₀, 1), y[1])
    Libtask.produce(score)

    for t in 2:(model.θ.T)
        state = rand(rng, f(model, model.X[t - 1], t - 1))
        model.X[t] = state
        score = logpdf(g(model, state, t), y[t])
        Libtask.produce(score)
    end
end

Libtask.tape_copy(model::NonLinearTimeSeries) = deepcopy(model)

Random.seed!(rng, seed)
model = NonLinearTimeSeries(θ₀)
pgas = AdvancedPS.PG(Nₚ)
chains = sample(rng, model, pgas, Nₛ; progress=false)

# Utility to replay a particle trajectory
function replay(particle::AdvancedPS.Particle)
    trng = deepcopy(particle.rng)
    Random123.set_counter!(trng.rng, 0)
    trng.count = 1
    model = NonLinearTimeSeries(θ₀)
    trace = AdvancedPS.Trace(AdvancedPS.GenericModel(model, trng), trng)
    score = AdvancedPS.advance!(trace, true)
    while !isnothing(score)
        score = AdvancedPS.advance!(trace, true)
    end
    return trace
end

trajectories = map(chains) do sample
    replay(sample.trajectory)
end

particles = hcat([trajectory.model.f.X for trajectory in trajectories]...) # Concat all sampled states
mean_trajectory = mean(particles; dims=2)

scatter(particles; label=false, opacity=0.01, color=:black)
plot!(x; color=:red, label="Original Trajectory")
plot!(mean_trajectory; color=:orange, label="Mean trajectory", opacity=0.9)
xlabel!("t")
ylabel!("State")

update_rate = sum(abs.(diff(particles; dims=2)) .> 0; dims=2) / Nₛ
plot(update_rate; label=false, ylim=[0, 1], legend=:bottomleft)
hline!([1 - 1 / Nₚ]; label="N: $(Nₚ)")
xlabel!("Iteration")
ylabel!("Update rate")
