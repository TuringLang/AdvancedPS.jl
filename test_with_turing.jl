include("Turing.jl/src/Turing.jl")
using Distributions

## It works even for GPU's!
cache = Dict()

Turing.@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))


  x ~ Normal(m, sqrt(s))
  h ~ Normal(m,abs(x))

  y ~ Normal(m, sqrt(s))

  k~ Exponential(0.1)
end

c1 = Turing.sample(gdemo(1.5, 2), Turing.PG(10), 10)
c2 = Turing.sample(gdemo(1.5, 2), Turing.SMC(), 10)
c3 = Turing.sample(gdemo(1.5,2), Turing.PGAS(10),10)
