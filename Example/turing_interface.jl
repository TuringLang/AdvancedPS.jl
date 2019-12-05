# This is important for initalizaiton
const initialize = AdvancedPS.current_trace
const TypedVarInfo = VarInfo{<:NamedTuple}

function report_observation!(trace, logp::Float64)
    trace.taskinfo.logp += logp
    produce(logp)
    trace = AdvancedPS.current_trace()
end

# logγ corresponds to the proposal distributoin we are sampling from.
function report_transition!(trace, logp::Float64, logγ::Float64)
    trace.taskinfo.logp += logp - logγ
    trace.taskinfo.logpseq += logp
end


# We obtain the new value for our variable
# If the distribution is not specified, we simply set it to be Normal
function update_var(trace, vn, val, dist= Normal())
    # check if the symbol is contained in the varinfo...
    if haskey(trace.vi,vn)
        if is_flagged(trace.vi, vn, "del")
            unset_flag!(trace.vi, vn, "del")
            trace.vi[vn] = vectorize(dist,val)
            setgid!(trace.vi, vn)
            setorder!(trace.vi, vn, trace.vi.num_produce)
            return val
        else
            val =  trace.vi[vn]
        end
    else
        #We do not specify the distribution... Thats why we set it to be Normal()
        push!(trace.vi, vn, val, dist)
    end
    return val
end

# The reason for this is that we need to pass it!
function create_task(f::Function)
    return CTask(() ->  begin new_vi=f(); produce(Val{:done}); new_vi; end )
end



#########################################################################
# This is copied from turing compiler3.0, but we need to extract the    #
# sampler form set_retained_vns_del_by_spl!                             #
#########################################################################


# Get all indices of variables belonging to SampleFromPrior:

#   if the gid/selector of a var is an empty Set, then that var is assumed to be assigned to

#   the SampleFromPrior sampler


"""
`getidx(vi::UntypedVarInfo, vn::VarName)`


Returns the index of `vn` in `vi.metadata.vns`.

"""
getidx(vi::UntypedVarInfo, vn::VarName) = vi.idcs[vn]


"""

`getidx(vi::TypedVarInfo, vn::VarName{sym})`

Returns the index of `vn` in `getfield(vi.metadata, sym).vns`.

"""

function getidx(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).idcs[vn]
end

setgid!(vi::UntypedVarInfo, vn::VarName) = push!(vi.gids[getidx(vi, vn)], Turing.Selector(1,:PS))

function setgid!(vi::TypedVarInfo, vn::VarName{sym}) where sym
    push!(getfield(vi.metadata, sym).gids[getidx(vi, vn)], Turing.Selector(1,:PS))
end


@inline function _getidcs(vi::UntypedVarInfo)
    return filter(i -> isempty(vi.gids[i]) , 1:length(vi.gids))
end

# Get a NamedTuple of all the indices belonging to SampleFromPrior, one for each symbol
@inline function _getidcs(vi::TypedVarInfo)
    return _getidcs(vi.metadata)
end

@generated function _getidcs(metadata::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = findinds(metadata.$f)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end







"""
`set_retained_vns_del_by_spl!(vi::VarInfo, spl::Sampler)`

Sets the `"del"` flag of variables in `vi` with `order > vi.num_produce` to `true`.

"""

function set_retained_vns_del_by_spl!(vi::UntypedVarInfo)
    # Get the indices of `vns` that belong to `spl` as a vector
    gidcs = _getidcs(vi)
    if vi.num_produce == 0
        for i = length(gidcs):-1:1
          vi.flags["del"][gidcs[i]] = true
        end
    else
        for i in 1:length(vi.orders)
            if i in gidcs && vi.orders[i] > vi.num_produce
                vi.flags["del"][i] = true
            end
        end
    end
    return nothing
end

function set_retained_vns_del_by_spl!(vi::TypedVarInfo)
    # Get the indices of `vns` that belong to `spl` as a NamedTuple, one entry for each symbol
    gidcs = _getidcs(vi)
    return _set_retained_vns_del_by_spl!(vi.metadata, gidcs, vi.num_produce)
end

@generated function _set_retained_vns_del_by_spl!(metadata, gidcs::NamedTuple{names}, num_produce) where {names}
    expr = Expr(:block)
    for f in names
        f_gidcs = :(gidcs.$f)
        f_orders = :(metadata.$f.orders)
        f_flags = :(metadata.$f.flags)
        push!(expr.args, quote
            # Set the flag for variables with symbol `f`
            if num_produce == 0
                for i = length($f_gidcs):-1:1
                    $f_flags["del"][$f_gidcs[i]] = true
                end
            else
                for i in 1:length($f_orders)
                    if i in $f_gidcs && $f_orders[i] > num_produce
                        $f_flags["del"][i] = true
                    end
                end
            end
        end)
    end
    return expr
end
