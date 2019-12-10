
@model gdemo_d() = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

gdemo_default = gdemo_d()

function gdemo_d_apf()
    var = initialize()
    r = rand(InverseGamma(2, 3))
    vn = @varname s
    s = update_var!(var, vn, r)

    r =  rand(Normal(0, sqrt(s)))
    vn = @varname m
    m = update_var!(var, vn, r)

    logp = logpdf(Normal(m, sqrt(s)), 1.5)
    report_observation!(var,logp)

    logp = logpdf(Normal(m, sqrt(s)), 2.0)
    report_observation!(var,logp)
end


aps_gdemo_default = gdemo_d_apf

@model large_demo(y) = begin
    x = TArray{Float64}(undef,11)
    x[1] ~ Normal()
    for i = 2:11
        x[i] ~ Normal(0.1*x[i-1]-0.1,0.5)
        y[i-1] ~ Normal(x[i],0.3)
    end
end

function large_demo_apf(y)
    var = initialize()
    x = TArray{Float64}(undef,11)
    vn = @varname x[1]
    x[1] = update_var!(var, vn, rand(Normal()))
    # there is nothing to report since we are not using proposal sampling
    report_transition!(var,0.0,0.0)
    for i = 2:11
        # Sampling
        r =  rand( Normal(0.1*x[i-1]-0.1,0.5))
        vn = @varname x[i]
        x[i] = update_var!(var, vn, r)
        logγ = logpdf( Normal(0.1*x[i-1]-0.1,0.5),x[i]) #γ(x_t|x_t-1)
        logp = logγ             # p(x_t|x_t-1)
        report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i], 0.3), y[i-1])
        var = report_observation!(var,logpy)
    end
end

## Here, we even have a proposal distributin
function large_demo_apf_proposal(y)
    var = initialize()
    x = TArray{Float64}(undef,11)
    vn = @varname x[1]
    x[1] = update_var!(var, vn, rand(Normal()))
    # there is nothing to report since we are not using proposal sampling
    report_transition!(var,0.0,0.0)
    for i = 2:11
        # Sampling
        r =  rand(Normal(0.1*x[i-1],0.5))
        vn = @varname x[i]
        x[i] = update_var!(var, vn, r)
        logγ = logpdf(Normal(0.1*x[i-1],0.5),x[i]) #γ(x_t|x_t-1)
        logp = logpdf( Normal(0.1*x[i-1]-0.1,0.5),x[i])             # p(x_t|x_t-1)
        report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i], 0.3), y[i-1])
        var = report_observation!(var,logpy)
    end
end
