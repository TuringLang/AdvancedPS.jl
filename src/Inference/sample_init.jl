function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    ℓ::ModelType,
    spl::SamplerType,
    N::Integer;
    debug::Bool=false,
    kwargs...
) where {ModelType<:AbstractPFModel, SamplerType<:SMCSampler}

    T = Trace{typeof(spl.vi),SMCTaskInfo{Float64}}
    particles = T[ get_new_trace(spl.vi, ℓ.task, SMCTaskInfo()) for _ =1:N]
    spl.pc = ParticleContainer{typeof(particles[1])}(particles,zeros(N),0.0,0)

    sample!(spl.pc, spl.alg, spl.uf, nothing)

end
