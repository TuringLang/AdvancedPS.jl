module AdvancedPS
using Libtask
using Random
using AbstractMCMC: AbstractSampler
using DynamicPPL: AbstractVarInfo, Model, SampleFromPrior, Sampler, reset_num_produce!, set_retained_vns_del_by_spl!, increment_num_produce!, getlogp, resetlogp!
using Distributions
using StatsFuns: softmax, logsumexp


# TODO: trace.jl no longer in AdvancedPS.jl package, must import Turing.jl/src/core/trace.jl
import Turing.Core: Trace, fork, forkr

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
        resample,
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