

const PROGRESS = Ref(true)

function AbstractMCMC.sample(
    model::ModelType,
    alg::AlgType,
    uf::UF,
    vi::C,
    N::Integer;
    kwargs...
) where {
    C,
    ModelType<:AbstractPFModel,
    AlgType<: AbstractPFAlgorithm,
    UF<:AbstractPFUtilitFunctions
}
    return sample(model, Sampler(alg, uf, vi), N; progress=PROGRESS[], kwargs...)
end

const ACVANCEDPS_INTERNAL_VARS = (internals = [
    "lp",
    "weight",
    "le",
],)

function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::AbstractPFModel,
    spl::AbstractPFSampler,
    N::Integer,
    ts::Vector{T};
    kwargs...
) where {T<:AbstractPFTransition}

    # Convert transitions to array format.
    # Also retrieve the variable names.
    nms, vals = _params_to_array(ts)

    # Get the values of the extra parameters in each Transition struct.

    extra_params, extra_values = get_transition_extras(ts)
    # Extract names & construct param array.
    nms = string.(vcat(nms..., string.(extra_params)...))
    parray = hcat(vals, extra_values)


    le = missing
    # Set up the info tuple.
    info = NamedTuple()
    # Chain construction.
    return Chains(
        parray,
        string.(nms),
        deepcopy(ACVANCEDPS_INTERNAL_VARS);
        evidence=le,
        info=info
    )
end
