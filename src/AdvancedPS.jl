module AdvancedPS
using Libtask
using Random
using AbstractMCMC: AbstractSampler
using DynamicPPL: AbstractVarInfo, Model, SampleFromPrior, Sampler, reset_num_produce!, set_retained_vns_del_by_spl!, increment_num_produce!, getlogp, resetlogp!
using Distributions
using StatsFuns: softmax, logsumexp

include("trace.jl")
export  Trace,
        fork,
        forkr,
        current_trace

include("particlecontainer.jl")
export  ParticleContainer, 
        reset_logweights!,
        increase_logweight!,
        getweights,
        getweight,
        logZ,
        effectiveSampleSize

include("resampling.jl")
export  ResampleWithESSThreshold,
        randcat,
        resample_multinomial,
        resample_residual,
        resample_stratified,
        resample_systematic

include("sweep.jl")
export  resample_propagate!,
        reweight!,
        sweep!

end
