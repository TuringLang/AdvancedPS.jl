# Import packages.
using Turing
using  Random; Random.seed!(1)
using Distributed
using DistributionsAD
# Define a simple Normal model with unknown mean and variance.
@model gdemo(y) = begin
  s ~ Exponential(0.2)
  m ~ Normal(0, sqrt(s))
  x = Vector{Real}(undef,10)
  x[1] ~ Normal(m,s)
  y[1] ~ Normal(x[1],0.5)
  for i = 2:10
    x[i] ~ Normal(x[i-1],s)
    y[i] ~ Normal(x[i],0.5)
  end

end

y = Vector{Float64}(1:10)

chn1 = sample(gdemo(y),SMC(),1000)
write("Old_Model_SMC.jls", chn1)
chn2 = sample(gdemo(y),PG(100),100)
write("Old_Model_PG.jls",chn2)
