

function sampleSMC!(pc::ParticleContainer,resampler::Function =resample_systematic ,resampler_threshold::AbstractFloat = 0.5)
    println("Hmmm")

    while consume(pc) != Val{:done}
        ess = effectiveSampleSize(pc)
        if ess <= resampler_threshold * length(pc)
            # compute weights
            Ws = weights(pc)
            # check that weights are not NaN
            @assert !any(isnan, Ws)
            # sample ancestor indices
            n = length(pc)
            nresamples = n
            indx = resampler(Ws, nresamples)
            resample!(pc, indx)
        end
    end
end

function samplePG!(pc::ParticleContainer,resampler::Function = resample_systematic ,ref_particle::Union{Particle,Nothing}=nothing , resampler_threshold =1)

    if ref_particle === nothing
        #Do normal SMC
        #By defualt, resampler_threshold is set to one, so that we always resample
        sampleSMC!(pc,resampler_threshold)
    else
        while consume(pc) != Val{:done}
            # compute weights
            Ws = weights(pc)
            # check that weights are not NaN
            @assert !any(isnan, Ws)
            # sample ancestor indices
            n = length(pc)
            # Ancestor trajectory is not sampled
            nresamples = n-1
            indx = resampler(Ws, nresamples)
            # We add ancestor trajectory to the path.
            # For ancestor sampling, we would change n at this point.
            push!(indx,n)
            resample!(pc, indx,ref_particle)
        end
    end

end
