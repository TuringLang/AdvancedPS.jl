include("Turing.jl/src/Turing.jl")
using Distributions
using CuArrays
## It works even for GPU's!
cache = Dict()

arr = rand(10,10)
arr2 = rand(10,1)
Turing.@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))


  arr3 = arr*arr2

  x ~ Normal(m, sqrt(s))
  h ~ Normal(m,abs(x))

  y ~ Normal(m, sqrt(s))

  k~ Exponential(0.1)
end

# arrcu = cu(rand(10000,10000))
# arr2cu = cu(rand(10000,1))
# Turing.@model gdemocu(x, y) = begin
#   s ~ InverseGamma(2, 3)
#   m ~ Normal(0, sqrt(s))
#
#
#   arr3cu = arrcu*arr2cu
#
#   x ~ Normal(m, sqrt(s))
#   h ~ Normal(m,abs(x))
#
#   y ~ Normal(m, sqrt(s))
#
#   k~ Exponential(0.1)
#
# end

#@elapsed arr*arr2
#@elapsed arrcu*arr2cu
#time2 = @elapsed c1 = Turing.sample(gdemocu(1.5, 2), Turing.PG(100), 100)

#time = @elapsed c1 = Turing.sample(gdemo(1.5, 2), Turing.PG(10), 10)

#c2 = Turing.sample(gdemo(1.5, 2), Turing.SMC(), 10)



@elapsed c1 = Turing.sample(gdemo(1.5,2), Turing.PGAS(10),100)
@elapsed c2 = Turing.sample(gdemo(1.5,2), Turing.PG(10),100)
@elapsed c3 = Turing.sample(gdemo(1.5,2), Turing.SMC(),1000)
