# We need to have the ParticleGibbsExtension branch!!!
using Turing
using Distributions
using CuArrays
## It works even for GPU's!
cache = Dict()

arr = rand(10,10)
arr2 = rand(10,1)
Turing.@model gdemo(x1,x2,x3,x4) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))


  arr3 = arr*arr2

  x1 ~ Normal(m, abs(s))
  h1 ~ Normal(m,abs(s))
  x2 ~ Normal(m, abs(h1))
  h2 ~ Normal(m,abs(h1))
  x3 ~ Normal(m, abs(h2))
  h3 ~ Normal(m,abs(h2))
  x4 ~ Normal(m, abs(h3))

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



@elapsed c1 = Turing.sample(gdemo([1.5,2,2,2]...), Turing.PGAS(100),1000)
@elapsed c2 = Turing.sample(gdemo([1.5,2,2,2]...), Turing.PG(100),1000)
@elapsed c3 = Turing.sample(gdemo([1.5,2,2,2]...), Turing.SMC(),1000)
@elapsed c4 = Turing.sample(gdemo([1.5,2,2,2]...), Turing.MH(),100000)

c2
c1
c3
c4
