module AdvancedSMC
import  Libtask,
        Random # doesn't seem like there is anything else to import? DynamicPPL and AbstractMCMC interfacing is handled in inference/

# must I export overloaded Base functions?
# perhaps: Inference.jl exports overloaded DynamicPPL functions 

include("particlecontainer.jl")
export  ParticleContainer, 
        reset_logweights!,
        increase_logweight!,
        getweights,
        getweight,
        logZ,
        effectiveSampleSize

include("sweep.jl")
export  resample_propagate!,
        reweight!,
        sweep!

include("resampling.jl")
export  ResampleWithESSThreshold,
        resample,
        randcat,
        resample_multinomial,
        resample_residual,
        resample_stratified,
        resample_systematic

include("trace.jl")
export  Trace,
        fork,
        forkr,
        current_trace,
        Particle
   
end