


struct PFTransition{T, F<:AbstractFloat} <: AbstractPFTransition
    θ::T
    lp::F
    le::F
    weight::F
end

AbstractMCMC.transition_type(spl::S) where S<:AbstractPFSampler = PFTransition
function additional_parameters(::Type{<:PFTransition})
    return [:lp,:le, :weight]
end



function get_transition_extras(ts::Vector{T}) where T<:AbstractTransition
    # Get the extra field names from the sampler state type.
    # This handles things like :lp or :weight.
    extra_params = additional_parameters(T)
    # Get the values of the extra parameters.
    local extra_names
    all_vals = []
    # Iterate through each transition.
    for t in ts
        extra_names = String[]
        vals = []
        # Iterate through each of the additional field names
        # in the struct.
        for p in extra_params
            # Check whether the field contains a NamedTuple,
            # in which case we need to iterate through each
            # key/value pair.
            prop = getproperty(t, p)
            if prop isa NamedTuple
                for (k, v) in pairs(prop)
                    push!(extra_names, string(k))
                    push!(vals, v)
                end
            else
                push!(extra_names, string(p))
                push!(vals, prop)
            end
        end
        push!(all_vals, vals)
    end
    # Convert the vector-of-vectors to a matrix.
    valmat = [all_vals[i][j] for i in 1:length(ts), j in 1:length(all_vals[1])]
    return extra_names, valmat
end





function flatten(names, value :: AbstractArray, k :: String, v)
    if isa(v, Number)
        name = k
        push!(value, v)
        push!(names, name)
    elseif isa(v, Array)
        for i = eachindex(v)
            if isa(v[i], Number)
                name = string(ind2sub(size(v), i))
                name = replace(name, "(" => "[");
                name = replace(name, ",)" => "]");
                name = replace(name, ")" => "]");
                name = k * name
                isa(v[i], Nothing) && println(v, i, v[i])
                push!(value, v[i])
                push!(names, name)
            elseif isa(v[i], AbstractArray)
                name = k * string(ind2sub(size(v), i))
                flatten(names, value, name, v[i])
            else
                error("Unknown var type: typeof($v[i])=$(typeof(v[i]))")
            end
        end
    else
        error("Unknown var type: typeof($v)=$(typeof(v))")
    end
    return
end

function flatten_namedtuple(nt::NamedTuple{pnames}) where {pnames}
    vals = Vector{Real}()
    names = Vector{AbstractString}()
    for k in pnames
        v = nt[k]
        if length(v) == 1
            flatten(names, vals, string(k), v)
        else
            for (vnval, vn) in zip(v[1], v[2])
                flatten(names, vals, vn, vnval)
            end
        end
    end
    return names, vals
end




function _params_to_array(ts::Vector{T}) where {T<:AbstractTransition}
    names = Set{String}()
    dicts = Vector{Dict{String, Any}}()
    # Extract the parameter names and values from each transition.
    for t in ts
        nms, vs = flatten_namedtuple(t.θ)
        if length(nms) >0
            push!(names, nms...)
            # Convert the names and values to a single dictionary.
            d = Dict{String, Any}()
            for (k, v) in zip(nms, vs)
                d[k] = v
            end
            push!(dicts, d)
        end
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

ind2sub(v, i) = Tuple(CartesianIndices(v)[i])
