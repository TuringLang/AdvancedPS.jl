### Attention this is a development package! It wount run.


module AdvancedPS



        using Libtask
        using StatsFuns: logsumexp, softmax!
        import Base.copy

        abstract type AbstractTaskInfo end
        abstract type AbstractParticleContainer end
        abstract type AbstractTrace end
        abstract type AbstractPFAlgorithm end
        abstract type AbstractPFUtilitFunctions end

        abstract type AbstractSMCUtilitFunctions <: AbstractPFUtilitFunctions end
        abstract type AbstractPGASUtilityFunctions <: AbstractSMCUtilitFunctions end


        include("UtilityFunctions.jl")
        include("trace.jl")
        include("ParticleContainer.jl")
        include("resample.jl")
        include("taskinfo.jl")
        include("samplers.jl")
        include("resample_functions.jl")


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



end # module
