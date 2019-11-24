using("Turing.jl")
using LinearAlgebra
using CUDAdrv
using CuArrays
# We use a very easy markovian model to test...
@model gdemo(y) = begin
  # Set priors.
  s ~ InverseGamma(2, 3)
  m ~ Normal(0,5)
  x = Vector{Real}(undef, 10)
  z = Vector{Real}(undef, 10)
  x[1] ~ Normal(0,1)
  for j = 2:10
    x[j] ~ Normal(m+x[j-1], sqrt(s))
    x[j] ~ Normal(x[j-1], sqrt(s))
    z[j] = x[j]^2
    y[j-1] ~ Normal(x[j],0.1)
  end
  return z
end

y = [ 1 2 3 4 5 6 7  8 9 10]
c = sample(gdemo(y), Gibbs(HMC(0.2,10,:m,:s),MH(:x)), 100)
sample(gdemo(y), SMC(:x,:y), 100)


hi = (4,"a")

arr = Vector{Symbol}(undef,1)
typeof((Array{Symbol,1},"a"))
isempty(arr)

Float64[]




v = rand(Normal())
pdf = logpdf(Normal(),v)


vectorize(Normal(),v)

dist =MvNormal([ 2. 1 1; 1 2 1; 1 1 2])

dist2 = Normal(2,3)

std(dist2)

mean(dist2)



s = []
for i = 1:10
  push!(s,i)
end

function pr(x)
  println(x)
end


M = rand(InverseWishart(100,Matrix{Float64}(I, 100, 100)))






@model gdemo() = begin
  # Set priors.
  pr("start")
  s ~ InverseGamma(2, 3)
  pr("in the first block")
  cholesky(M)
  m ~ Normal(0,5)
  pr("hi n the block")
  x = Normal(0, 10)
  pr("end")
  return x
end

sample(gdemo(), Gibbs(HMC(0.2,1,:m,:s),MH(:x)), 2)



push!(LOAD_PATH, string(pwd(),"/Turing/src/"))
push!(LOAD_PATH, string(pwd(),"/AdvancedPS/src/"))
using Pkg
include(string(pwd(),"/Turing/src/Turing.jl"))
include(string(pwd(),"/AdvancedPS/src/AdvancedPS.jl"))

using AdvancedPS
using Turing

@model mld = begin
  x ~Normal()
end


F = (copy = copy, pr = println)
F[:pr](4)
typeof(F) <: NamedTuple{}

NamedTuple{}()

names(F)
@assert (:copy,:pr) in keys(F)
