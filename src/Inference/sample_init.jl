function sample_init!(
    rng::AbstractRNG,
    ℓ::ModelType,
    s::SamplerType,
    N::Integer;
    debug::Bool=false,
    kwargs...
) where {ModelType<:AbstractPFModel, SamplerType<:SMCSampler}

    T = Trace{typeof(spl.vi),SMCTaskInfo}
    particles = T[ Trace(vi, task, SMCTaskInfo(), alg.copy) for _ =1:N]
    spl.pc = APS.ParticleContainer{typeof(particles[1])}(particles,zeros(N),zeros(N),0,0)

    sample!(spl.pc, spl.alg, spl.uf, nothing)

end
