using Pkg
Pkg.activate("/sftwr/user-pkg/m1cak00/julia/dev/AdvancedPS/examples/gaussian-process")

# # Gaussian Process State-Space Model (GP-SSM)
using LinearAlgebra
using Random
using AdvancedPS
using AbstractGPs
using Plots
using Distributions
using Libtask
using SSMProblems

struct GaussianProcessDynamics{T<:Real,KT<:Kernel} <: LatentDynamics{T,T}
    proc::GP{ZeroMean{T},KT}
    function GaussianProcessDynamics(::Type{T}, kernel::KT) where {T<:Real,KT<:Kernel}
        return new{T,KT}(GP(ZeroMean{T}(), kernel))
    end
end

struct LinearGaussianDynamics{T<:Real} <: LatentDynamics{T,T}
    a::T
    b::T
    q::T
end

function SSMProblems.distribution(proc::LinearGaussianDynamics{T}) where {T<:Real}
    return Normal(zero(T), proc.q)
end

function SSMProblems.distribution(proc::LinearGaussianDynamics, ::Int, state)
    return Normal(proc.a * state + proc.b, proc.q)
end

struct StochasticVolatility{T<:Real} <: ObservationProcess{T,T} end

function SSMProblems.distribution(::StochasticVolatility{T}, ::Int, state) where {T<:Real}
    return Normal(zero(T), exp((1 / 2) * state))
end

function LinearGaussianStochasticVolatilityModel(a::T, q::T) where {T<:Real}
    dyn = LinearGaussianDynamics(a, zero(T), q)
    obs = StochasticVolatility{T}()
    return SSMProblems.StateSpaceModel(dyn, obs)
end

function GaussianProcessStateSpaceModel(::Type{T}, kernel::KT) where {T<:Real,KT<:Kernel}
    dyn = GaussianProcessDynamics(T, kernel)
    obs = StochasticVolatility{T}()
    return SSMProblems.StateSpaceModel(dyn, obs)
end

const GPSSM{T,KT<:Kernel} = SSMProblems.StateSpaceModel{
    T,
    GaussianProcessDynamics{T,KT},
    StochasticVolatility{T}
};

# for non-markovian models, we can redefine dynamics to reference the trajectory
function AdvancedPS.dynamics(
    ssm::AdvancedPS.TracedSSM{<:GPSSM{T},T,T}, step::Int
) where {T<:Real}
    prior = ssm.model.dyn.proc(1:(step - 1))
    post  = posterior(prior, ssm.X[1:(step - 1)])
    μ, σ  = mean_and_cov(post, [step])
    return LinearGaussianDynamics(zero(T), μ[1], sqrt(σ[1]))
end

# Everything is now ready to simulate some data. 
rng = MersenneTwister(1234);
true_model = LinearGaussianStochasticVolatilityModel(0.9, 0.5);
_, x, y = sample(rng, true_model, 100);

# Create the model and run the sampler
gpssm = GaussianProcessStateSpaceModel(Float64, SqExponentialKernel());
model = gpssm(y);
pg = AdvancedPS.PGAS(20);
chains = sample(rng, model, pg, 250; progress=false);
#md nothing #hide

particles = hcat([chain.trajectory.model.X for chain in chains]...);
mean_trajectory = mean(particles; dims=2);
#md nothing #hide

scatter(particles; label=false, opacity=0.01, color=:black, xlabel="t", ylabel="state")
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)
