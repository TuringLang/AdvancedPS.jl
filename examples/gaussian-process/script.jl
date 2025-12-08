# # Gaussian Process State-Space Model (GP-SSM)
using LinearAlgebra
using Random
using AdvancedPS
using AbstractGPs
using Plots
using Distributions
using Libtask
using SSMProblems

struct GaussianProcessDynamics{T<:Real,KT<:Kernel} <: SSMProblems.LatentDynamics
    proc::GP{ZeroMean{T},KT}
    function GaussianProcessDynamics(::Type{T}, kernel::KT) where {T<:Real,KT<:Kernel}
        return new{T,KT}(GP(ZeroMean{T}(), kernel))
    end
end

struct GaussianPrior{ΣT<:Real} <: SSMProblems.StatePrior
    σ::ΣT
end

SSMProblems.distribution(proc::GaussianPrior) = Normal(0, proc.σ)

struct LinearGaussianDynamics{AT<:Real,BT<:Real,QT<:Real} <: SSMProblems.LatentDynamics
    a::AT
    b::BT
    q::QT
end

function SSMProblems.distribution(proc::LinearGaussianDynamics, ::Int, state)
    return Normal(proc.a * state + proc.b, proc.q)
end

struct StochasticVolatility <: SSMProblems.ObservationProcess end

function SSMProblems.distribution(::StochasticVolatility, ::Int, state)
    return Normal(0, exp(state / 2))
end

function LinearGaussianStochasticVolatilityModel(a, q)
    prior = GaussianPrior(q)
    dyn = LinearGaussianDynamics(a, 0, q)
    obs = StochasticVolatility()
    return SSMProblems.StateSpaceModel(prior, dyn, obs)
end

function GaussianProcessStateSpaceModel(::Type{T}, kernel::KT) where {T<:Real,KT<:Kernel}
    prior = GaussianPrior(one(T))
    dyn = GaussianProcessDynamics(T, kernel)
    obs = StochasticVolatility()
    return SSMProblems.StateSpaceModel(prior, dyn, obs)
end

const GPSSM{T,KT<:Kernel} = SSMProblems.StateSpaceModel{
    <:GaussianPrior,<:GaussianProcessDynamics{T,KT},StochasticVolatility
};

# for non-markovian models, we can redefine dynamics to reference the trajectory
function AdvancedPS.dynamics(ssm::AdvancedPS.TracedSSM{<:GPSSM}, step::Int)
    prior = ssm.model.dyn.proc(1:(step - 1))
    post = posterior(prior, ssm.X[1:(step - 1)])
    μ, σ = mean_and_cov(post, [step])
    return LinearGaussianDynamics(0, μ[1], sqrt(σ[1]))
end

# Everything is now ready to simulate some data. 
rng = MersenneTwister(1234);
true_model = LinearGaussianStochasticVolatilityModel(0.9, 0.5);
_, x, y = sample(rng, true_model, 100);

# Create the model and run the sampler
gpssm = GaussianProcessStateSpaceModel(Float64, SqExponentialKernel());
model = AdvancedPS.TracedSSM(gpssm, y);
pg = AdvancedPS.PGAS(20);
chains = sample(rng, model, pg, 250);
#md nothing #hide

particles = hcat([chain.trajectory.model.X for chain in chains]...);
mean_trajectory = mean(particles; dims=2);
#md nothing #hide

scatter(particles; label=false, opacity=0.01, color=:black, xlabel="t", ylabel="state")
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)
