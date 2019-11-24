
function resample!(
    pc :: ParticleContainer,
    indx :: Vecotr{Int64} = resample_systematic,
    ref :: Union{Particle, Nothing} = nothing
    ancestor_idx :: int = -1
    new_ancestor_traj :: Union{Particle,Nothing} = nothing
)


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
            p = isref ? fork(pi,  pc.mainpulators["copy"], isref, pc.mainpulators["set_retained_vns_del_by_spl!"]) : pi
            children[j += 1] = p

            # fork additional children
            for _ in 2:ni
                children[j += 1] = fork(p, pc.mainpulators["copy"], isref, pc.mainpulators["set_retained_vns_del_by_spl!"])
            end
        end
    end

    if ref !== nothing
        # Insert the retained particle. This is based on the replaying trick for efficiency
        # reasons. If we implement PG using task copying, we need to store Nx * T particles!
        # This is a rather effcient way of how to solve the ancestor problem.
        if ancestor_idx == n || ancestor_idx == -1
            @inbounds children[n] = ref
        else
            @assert isa(Particle,new_ancestor_traj) "[AdvancedPS] ($new_ancestor_traj) must be of type particle"
            @inbounds children[n] = new_ancestor_traj
        else

    end

    # replace particles and log weights in the container with new particles and weights
    pc.vals = children
    pc.logWs = zeros(n)

    pc
end
