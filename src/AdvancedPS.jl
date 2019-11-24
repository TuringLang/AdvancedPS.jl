### Attention this is a development package! It wount run.


module AdvancedPS



        const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_APS", "0")))
        using Libtask
        using StatsFuns: logsumexp, softmax!
        import Base.copy

        abstract type AbstractTaskInfo end
        abstract type AbstractParticleContainer end
        abstract type AbstractTrace end


        include("tasks.jl")

        export  ParticleContainer,
                Trace,
                weights,
                logZ,
                current_trace,
                extend!


        include("resample.jl")

        export resample!




        include("taskinfo.jl")

        export PGTaskInfo

        include("samplers.jl")

        export sampleSMC!, samplePG!


        include("resample_functions.jl")

        export  resample,
                randcat,
                resample_multinomial,
                resample_residual,
                resample_stratified,
                resample_systematic

        include("Interface.jl")

        export  report_transition!,
                report_observation!,
                init,
                Container,
                set_x,
                get_x,
                copyC,
                create_task

end # module
