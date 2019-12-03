# Import packages.

using Turing
using  Random; Random.seed!(1)
using Distributed
using DistributionsAD
# Define a simple Normal model with unknown mean and variance.
@Turing.model gdemo(y) = begin
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

chn1 = Turing.sample(gdemo(y),Turing.SMC(),1000)
write("tmp.jls", chn1)
chn2 = Turing.sample(gdemo(y),Turing.PG(100),100)
write("tmp2.jls",chn2)

chn1_old = read("Old_Model_SMC.jls", Chains)
chn2_old = read("Old_Model_PG.jls", Chains)

sum(chn1.value.data - chn1_old.value.data)
sum(chn2.value.data- chn2_old.value.data)
