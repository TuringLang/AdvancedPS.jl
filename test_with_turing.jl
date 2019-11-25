include("Turing.jl/src/Turing.jl")
using Distributions
using CuArrays

## It works even for GPU's!
cache = Dict()

Turing.@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  if :x in spl.space

  @cond_compute cache (:x,:s,:m,:h) begin
    arr = cu(rand(1000,1000))
    arr2 = cu(rand(1000,1))
    cache[:arr] = arr*arr2
    arr = cache[:arr]

    x ~ Normal(m, sqrt(s))
    h ~ Normal(m,abs(x))
  end


  :x in spl.space ? ... : cache[:x]

  y ~ Normal(m, sqrt(s))

  k~ Exponential(0.1)
end

c1 = Turing.sample(gdemo(1.5, 2), Turing.PG(100), 1000)
c2 = Turing.sample(gdemo(1.5, 2), Turing.SMC(), 10)
