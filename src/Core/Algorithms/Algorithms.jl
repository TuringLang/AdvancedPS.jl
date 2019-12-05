
struct SMCAlgorithm{RT} <: AbstractPFAlgorithm where RT<:AbstractFloat
    resampler             ::  Function
    resampler_threshold   ::  RT

end

function SMCAlgorithm()
    SMCAlgorithm(resample_systematic, 0.5)
end

struct PGAlgorithm{RT} <: AbstractPFAlgorithm where RT<:AbstractFloat
    resampler             ::  Function
    resampler_threshold   ::  RT
    n                     ::  Int64

end

function PGAlgorithm(n::Int64)
    PGAlgorithm(resample_systematic, 0.5, n)
end


struct PGASAlgorithm{RT} <: AbstractPFAlgorithm
    resampler             ::  Function
    resampler_threshold   ::  RT
    merge_traj            ::  Function
    n                     ::  Int64
end

function PGASAlgorithm(merge_traj::Function, n::Int64)
    SMCAlgorithm(resample_systematic, 0.5, merge_traj, n)
end
