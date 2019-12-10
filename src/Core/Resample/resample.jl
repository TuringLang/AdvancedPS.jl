
function resample!(
    pc :: ParticleContainer,
    utility_functions :: AbstractSMCUtilitFunctions,
    indx :: Vector{Int64},
    ref :: Union{Particle, Nothing} = nothing,
    new_ref:: Union{Particle, Nothing} = nothing
)

    n = length(pc.vals)
    # count number of children for each particle
    num_children = zeros(Int, n)

    @inbounds for i in indx
        num_children[i] += 1
    end



    # fork particles
    particles = collect(pc)
    children = similar(particles)
    j = 0
    @inbounds for i in 1:n
        ni = num_children[i]

        if ni > 0
            # fork first child
            pi = particles[i]
            isref = pi === ref
            p = isref ? fork(pi, isref, utility_functions.set_retained_vns_del_by_spl!) : pi
            children[j += 1] = p

            # fork additional children
            for _ in 2:ni
                children[j += 1] = fork(p,  isref, utility_functions.set_retained_vns_del_by_spl!)
            end
        end
    end

    if ref !== nothing
        ancestor_idx = indx[n]
        # Insert the retained particle. This is based on the replaying trick for efficiency
        # reasons. If we implement PG using task copying, we need to store Nx * T particles!
        # This is a rather effcient way of how to solve the ancestor problem.
        if ancestor_idx == n
            @inbounds children[n] = ref
        elseif new_ref !== nothing
            @inbounds children[n] = new_ref
        end
    end

    # replace particles and log weights in the container with new particles and weights
    pc.vals = children
    pc.logWs = zeros(n)

    pc
end
