### Attention this is a development package! It wount run.


module AdvancedPS


        using Libtask
        using StatsFuns: logsumexp, softmax!
        using AdvancedMCMC
        import MCMCChains: Chains
        import Base.copy

        abstract type AbstractTaskInfo end
        abstract type AbstractParticleContainer end
        abstract type AbstractTrace end
        abstract type AbstractPFAlgorithm end
        abstract type AbstractPFUtilitFunctions end

        abstract type AbstractPFTransition <: AbstractTransition end
        abstract type AbstractPFSampler <: AbstractSampler end

        abstract type AbstractSMCUtilitFunctions <: AbstractPFUtilitFunctions end
        abstract type AbstractPGASUtilityFunctions <: AbstractSMCUtilitFunctions end

        abstract type AbstractPFModel <: AbstractModel end



        include("Core/Container/trace.jl")
        include("Core/Container/ParticleContainer.jl")
        include("Core/Resample/resample.jl")
        include("Core/Algorithms/taskinfo.jl")
        include("Core/Algorithms/sample.jl")
        include("Core/Resample/resample_functions.jl")
        include("Core/Algorithms/Algorithms.jl")
        include("Core/Utilities/UtilityFunctions.jl")

        export  ParticleContainer,
                Trace,
                weights,
                logZ,
                current_trace,
                extend!,
                empty!,
                resample!,
                PGTaskInfo,
                PGASTaskInfo
                sample!,
                resample,
                randcat,
                resample_multinomial,
                resample_residual,
                resample_stratified,
                resample_systematic,
                SMCUtilityFunctions,
                PGUtilityFunctions,
                PGASUtilityFunctions,
                SMCAlgorithm,
                PGAlgorithm

        include("Inference/Model.jl")
        include("Inference/sample_init.jl")
        include("Inference/Sampler.jl")
        inlcude("Inference/step.jl")
        include("Inference/Transitions.jl")
        include("Inference/Inference.jl")

        export  PFModel,
                sample,
                bundle_samples,
                sample_init!,
                step!,
                Sampler,
                PFTransition

end # module
