

## This is all we need for Turing!

function sample!(pc::PC, alg::ALG, utility_functions::AbstractSMCUtilitFunctions, ref_traj::T) where {
    PC <:ParticleContainer,
    ALG <:Union{SMCAlgorithm, PGAlgorithm},
    T <:Union{Trace,Nothing}
}
    n = length(pc.vals)
    while consume(pc) != Val{:done}
        ess = effectiveSampleSize(pc)
        if ref_traj !== nothing || ess <= alg.resampler_threshold * length(pc)
            # compute weights
            Ws = weights(pc)
            # check that weights are not NaN
            @assert !any(isnan, Ws)
            # sample ancestor indices
            # Ancestor trajectory is not sampled
            ref_traj !== nothing ? nresamples = n-1 : nresamples = n

            indx = alg.resampler(Ws, nresamples)
            # We add ancestor trajectory to the path.
            # For ancestor sampling, we would change n at this point.
            ref_traj !== nothing ? push!(indx,n) : nothing
            resample!(pc, utility_functions, indx, ref_traj)
        end
    end
end
