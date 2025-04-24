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

    test = ExactOneSampleKSTest(final_particles, Normal(Xf.x[end].μ, sqrt(Xf.x[end].Σ)))
    return pvalue(test)
end

@testset "linear-gaussian.jl" begin
    T = 3
    N_PARTICLES = 100
    N_SAMPLES = 200

    # Model dynamics
    a = 0.5
    b = 0.2
    q = 0.1
    E = LinearEvolution(a, Gaussian(b, q))

    H = 1.0
    R = 0.1
    Obs = LinearObservationModel(H, R)

    x0 = 0.0
    P0 = 1.0
    G0 = Gaussian(x0, P0)

    M = LinearStateSpaceModel(E, Obs)
    O = LinearObservation(E, H, R)

    # Simulate from model
    rng = StableRNG(1234)
    initial = rand(rng, StateObs(G0, M.obs))
    trajectory = trace(DynamicIterators.Sampled(M, rng), 1 => initial, endtime(T))
    y_pairs = collect(t => y for (t, (x, y)) in pairs(trajectory))
    ys = [y for (t, (x, y)) in pairs(trajectory)]

    # Ground truth smoothing
    Xf, ll = kalmanfilter(M, 1 => G0, y_pairs)

    # Define AdvancedPS model
    struct LinearGaussianDynamics{T<:Real} <: LatentDynamics{T,T}
        a::T
        b::T
        q::T
    end

    function SSMProblems.distribution(proc::LinearGaussianDynamics; kwargs...)
        return Normal(convert(T, x0), convert(T, p0))
    end

    function SSMProblems.distribution(
        proc::LinearGaussianDynamics, step::Int, state; kwargs...
    )
        return Normal(proc.a * state + proc.b, proc.q)
    end

    struct LinearGaussianObservation{T<:Real} <: ObservationProcess{T,T}
        h::T
        r::T
    end

    function SSMProblems.distribution(
        proc::LinearGaussianObservation, step::Int, state; kwargs...
    )
        return Normal(proc.h * state, proc.r)
    end

    function LinearGaussianStateSpaceModel(a, b, q, h, r)
        dyn = LinearGaussianDynamics(a, b, q)
        obs = LinearGaussianObservation(h, r)
        return StateSpaceModel(dyn, obs)
    end

    lgssm = LinearGaussianStateSpaceModel(a, b, q, H, R)
    model = lgssm(ys)

    @testset "PGAS" begin
        pgas = AdvancedPS.PGAS(N_PARTICLES)
        p = test_algorithm(rng, pgas, model, N_SAMPLES, Xf)
        @info p
        @test p > 0.05
    end

    @testset "PG" begin
        pg = AdvancedPS.PG(N_PARTICLES)
        p = test_algorithm(rng, pg, model, N_SAMPLES, Xf)
        @info p
        @test p > 0.05
    end
end
