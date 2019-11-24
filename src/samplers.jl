

function sampleSMC!(pc::ParticleContainer,resampler_threshold::AbstractFloat)
    while consume(pc) != Val{:done}
        ess = effectiveSampleSize(pc)
        if ess <= spl.alg.resampler_threshold * length(pc)
            # compute weights
            Ws = weights(pc)
            # check that weights are not NaN
            @assert !any(isnan, Ws)
            # sample ancestor indices
            n = length(pc)
            nresamples = n
            indx = randcat(Ws, nresamples)
            resample!(particles, indx)
        end
    end
end


function samplePG!(pc::ParticleContainer,ref_particle::Union{Particle,},resampler::Function )

    while consume(pc) != Val{:done}
        resample!(pc, resampler, ref_particle)
    end

end
