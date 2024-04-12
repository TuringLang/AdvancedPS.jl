"""
Unit tests for the validity of the SMC algorithms included in this package.

We test each SMC algorithm on a one-dimensional linear Gaussian state space model for which
an analytic filtering distribution can be computed using the Kalman filter provided by the
`Kalman.jl` package.

The validity of the algorithm is tested by comparing the final estimated filtering
distribution ground truth using a one-sided Kolmogorov-Smirnov test.
"""

using DynamicIterators
using GaussianDistributions
using HypothesisTests
using Kalman

function test_algorithm(rng, algorithm, model, N_SAMPLES, Xf)
    chains = sample(rng, model, algorithm, N_SAMPLES; progress=false)
    particles = hcat([chain.trajectory.model.X for chain in chains]...)
    final_particles = particles[:, end]

    test = ExactOneSampleKSTest(
        final_particles, Normal(Xf.x[end].μ[1], sqrt(Xf.x[end].Σ[1, 1]))
    )
    return pvalue(test)
end

@testset "linear-gaussian.jl" begin
    T = 3
    N_PARTICLES = 20
    N_SAMPLES = 50

    # Model dynamics (in matrix form, despite being one-dimensional, to work with Kalman.jl)
    Φ = [0.5;;]
    b = [0.2]
    Q = [0.1;;]
    E = LinearEvolution(Φ, Gaussian(b, Q))

    H = [1.0;;]
    R = [0.1;;]
    Obs = LinearObservationModel(H, R)

    x0 = [0.0]
    P0 = [1.0;;]
    G0 = Gaussian(x0, P0)

    M = LinearStateSpaceModel(E, Obs)
    O = LinearObservation(E, H, R)

    # Simulate from model
    rng = StableRNG(1234)
    initial = rand(rng, StateObs(G0, M.obs))
    trajectory = trace(DynamicIterators.Sampled(M), 1 => initial, endtime(T))
    y_pairs = collect(t => y for (t, (x, y)) in pairs(trajectory))
    ys = stack(y for (t, (x, y)) in pairs(trajectory))

    # Ground truth smoothing
    Xf, ll = kalmanfilter(M, 1 => G0, y_pairs)

    # Define AdvancedPS model
    mutable struct LinearGaussianParams
        a::Float64
        b::Float64
        q::Float64
        h::Float64
        r::Float64
        x0::Float64
        p0::Float64
    end

    mutable struct LinearGaussianModel <: AdvancedPS.AbstractStateSpaceModel
        X::Vector{Float64}
        θ::LinearGaussianParams
        LinearGaussianModel(params::LinearGaussianParams) = new(Vector{Float64}(), params)
    end

    function AdvancedPS.initialization(model::LinearGaussianModel)
        return Normal(model.θ.x0, model.θ.p0)
    end
    function AdvancedPS.transition(model::LinearGaussianModel, state, step)
        return Normal(model.θ.a * state + model.θ.b, model.θ.q)
    end
    function AdvancedPS.observation(model::LinearGaussianModel, state, step)
        return logpdf(Normal(model.θ.h * state, model.θ.r), ys[step])
    end

    AdvancedPS.isdone(::LinearGaussianModel, step) = step > T

    params = LinearGaussianParams(Φ[1, 1], b[1], Q[1, 1], H[1, 1], R[1, 1], x0[1], P0[1, 1])
    model = LinearGaussianModel(params)

    @testset "PGAS" begin
        pgas = AdvancedPS.PGAS(N_PARTICLES)
        p = test_algorithm(rng, pgas, model, N_SAMPLES, Xf)
        @test p > 0.05
    end

    @testset "PG" begin
        pg = AdvancedPS.PG(N_PARTICLES)
        p = test_algorithm(rng, pg, model, N_SAMPLES, Xf)
        @test p > 0.05
    end
end
