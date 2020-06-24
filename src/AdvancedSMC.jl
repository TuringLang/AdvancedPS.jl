module AdvancedSMC
using  Libtask,
        Random,
        AbstractMCMC
# must I export overloaded Base functions?
# perhaps: Inference.jl exports overloaded DynamicPPL functions 

include("trace.jl")
export  Trace,
        fork,
        forkr,
        current_trace,
        Particle
   

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