

function AbstractMCMC.sample(
    model::ModelType,
    alg::AlgType,
    uf::UF,
    vi::T,
    N::Integer;
    kwargs...
) where {
    ModelType<:AbstractPFModel,
    AlgType<: AbstractPFAlgorithm,
    UF<:AbstractPFUtilitFunctions
}
    return sample(model, Sampler(alg, uf, vi), N; progress=PROGRESS[], kwargs...)
end


function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::ModelType,
    spl::AbstractPFSampler,
    N::Integer,
    ts::Vector{T};
    discard_adapt::Bool=true,
    save_state=false,
    kwargs...
) where {ModelType<:AbstractPFModel, T<:AbstractPFTransition}

    # Convert transitions to array format.
    # Also retrieve the variable names.
    nms, vals = _params_to_array(ts)


    # Get the values of the extra parameters in each Transition struct.

    extra_params, extra_values = get_transition_extras(ts)
    # Extract names & construct param array.
    nms = string.(vcat(nms..., string.(extra_params)...))
    parray = hcat(vals, extra_values)


    # Get the values of the extra parameters in each Transition struct.
    extra_params, extra_values = get_transition_extras(ts)
    le = missing
    # Set up the info tuple.
    info = NamedTuple()
    # Chain construction.
    return Chains(
        parray,
        string.(nms),
        deepcopy(TURING_INTERNAL_VARS);
        evidence=le,
        info=info
    )
end


function _params_to_array(ts::Vector{T}) where {T<:AbstractTransition}
    names = Set{String}()
    dicts = Vector{Dict{String, Any}}()
    # Extract the parameter names and values from each transition.
    for t in ts
        nms, vs = flatten_namedtuple(t.θ)
        push!(names, nms...)
        # Convert the names and values to a single dictionary.
        d = Dict{String, Any}()
        for (k, v) in zip(nms, vs)
            d[k] = v
        end
        push!(dicts, d)
    end
    # Convert the set to an ordered vector so the parameter ordering
    # is deterministic.
    ordered_names = collect(names)
    vals = Matrix{Union{Real, Missing}}(undef, length(ts), length(ordered_names))
    # Place each element of all dicts into the returned value matrix.
    for i in eachindex(dicts)
        for (j, key) in enumerate(ordered_names)
            vals[i,j] = get(dicts[i], key, missing)
        end
    end
    return ordered_names, vals
end
