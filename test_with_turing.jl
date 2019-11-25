include("Turing.jl/src/Turing.jl")
using Distributions


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
