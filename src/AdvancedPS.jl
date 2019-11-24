### Attention this is a development package! It wount run.


module AdvancedPS



        const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_APS", "0")))





        import StatsBase: sample


        abstract type AbstractTaskInfo end

        include("resample.jl")

        export resample!


        include("tasks.jl")

        export ParticleContainer, Trace


        include("taskinfo.jl")

        export AbstractTaskInfo, TaskInfo

        include("samplers.jl")

        export sampleSMC!, samplePG!


        include("resample_functions.jl")

        export  resample,
                randcat,
                resample_multinomial,
                resample_residual,
                resample_stratified,
                resample_systematic



end # module
