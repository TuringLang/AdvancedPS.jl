
struct SMCAlgorithm{RT} <: AbstractPFAlgorithm where RT<:AbstractFloat
    resampler             ::  Function
    resampler_threshold   ::  RT

end

function SMCAlgorithm()
    SMCAlgorithm(resample_systematic, 0.5)
end


const PGAlgorithm = SMCAlgorithm


struct PGASAlgorithm{RT} <: AbstractPFAlgorithm
    resampler             ::  Function
    resampler_threshold   ::  RT
    merge_traj            ::  Function
    n                     ::  Int64
end

function PGASAlgorithm(merge_traj::Function, n::Int64)
    SMCAlgorithm(resample_systematic, 0.5, merge_traj, n)
end
