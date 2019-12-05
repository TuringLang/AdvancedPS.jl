using Turing

n = 20

y = Vector{Float64}(undef,n-1)
for i =1:n-1
    y[i] = sin(0.1*i)
end

@model demo(y) = begin
    x = Vector{Float64}(undef,n)
    vn = @varname x[1]
    x[1] ~ Normal()
    for i = 2:n
        x[i] ~ Normal(x[i-1],0.2)
        y[i-1] ~ Normal(x[i],0.2)
    end
end

sample(demo(y),SMC(),10)
