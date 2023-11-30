# # The Levy State Space Model (Godsill et al. 2020)

include("simulation.jl")

Parameters = @NamedTuple begin
    μ_W::Float64  # Subordinator process mean
    σ_W::Float64  # Subordinator process variance
    β::Float64    # Tempering parameter
    C::Float64    # Jump density
end

# Simulation parameters
T = 100    # Time horizon
res = 100  # Sampling resolution

# Sampling parameters




