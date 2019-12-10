module AdvancedPS_Turing_Container

    using Turing.Core.RandomVariables
    import Turing.Core:  @varname
    import Turing.Utilities: vectorize
    using Turing
    using AdvancedPS: current_trace

    # This is important for initalizaiton
    const TypedVarInfo = VarInfo{<:NamedTuple}
    const Selector = Turing.Selector
    const BASE_SELECTOR = Selector(:PS)
    const initialize = current_trace


    Base.copy(vi::VarInfo) = deepcopy(vi)
    tonamedtuple(vi::TypedVarInfo) = Turing.tonamedtuple(vi)
    tonamedtuple(vi::UntypedVarInfo) = tonamedtuple(TypedVarInfo(vi))


    function report_observation!(trace, logp::Float64)
        produce(logp)
        trace = current_trace()
    end

    # logγ corresponds to the proposal distributoin we are sampling from.
    function report_transition!(trace, logp::Float64, logγ::Float64)
        trace.taskinfo.logp += logp - logγ
        trace.taskinfo.logpseq += logp
    end


    # We obtain the new value for our variable
    # If the distribution is not specified, we simply set it to be Normal
    function update_var!(trace, vn, val, dist= Normal())
        # check if the symbol is contained in the varinfo...
        if haskey(trace.vi,vn)
            if is_flagged(trace.vi, vn, "del")
                unset_flag!(trace.vi, vn, "del")
                trace.vi[vn] = vectorize(dist,val)
                setgid!(trace.vi, BASE_SELECTOR, vn)
                setorder!(trace.vi, vn, trace.vi.num_produce)
                return val
            else
                updategid!(trace.vi, BASE_SELECTOR, vn)
                val =  trace.vi[vn]
            end
        else
            #We do not specify the distribution... Thats why we set it to be Normal()
            push!(trace.vi, vn, val, dist)
        end
        return val
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

    """
    `setgid!(vi::VarInfo, gid::Selector, vn::VarName)`

    Adds `gid` to the set of sampler selectors associated with `vn` in `vi`.
    """
    setgid!(vi::UntypedVarInfo, gid::Selector, vn::VarName) = push!(vi.gids[getidx(vi, vn)], gid)
    function setgid!(vi::TypedVarInfo, gid::Selector, vn::VarName{sym}) where sym
        push!(getfield(vi.metadata, sym).gids[getidx(vi, vn)], gid)
    end
    """
    `updategid!(vi::VarInfo, vn::VarName, spl::Sampler)`

    If `vn` doesn't have a sampler selector linked and `vn`'s symbol is in the space of
    `spl`, this function will set `vn`'s `gid` to `Set([spl.selector])`.
    """
    function updategid!(vi::AbstractVarInfo, sel::Selector, vn::VarName)
        setgid!(vi, sel, vn)
    end

    @generated function _getidcs(metadata::NamedTuple{names}) where {names}
        exprs = []
        for f in names
            push!(exprs, :($f = findinds(metadata.$f)))
        end
        length(exprs) == 0 && return :(NamedTuple())
        return :($(exprs...),)
    end

    # Get all indices of variables belonging to a given sampler

    @inline function _getidcs(vi::UntypedVarInfo, s::Selector)
        findinds(vi, s)
    end

    @inline function _getidcs(vi::TypedVarInfo, s::Selector)
        return _getidcs(vi.metadata, s)
    end
    # Get a NamedTuple for all the indices belonging to a given selector for each symbol
    @generated function _getidcs(metadata::NamedTuple{names}, s::Selector) where {names}
        exprs = []
        # Iterate through each varname in metadata.
        for f in names
            # If the varname is in the sampler space
            # or the sample space is empty (all variables)
            # then return the indices for that variable.
            push!(exprs, :($f = findinds(metadata.$f, s)))
        end
        length(exprs) == 0 && return :(NamedTuple())
        return :($(exprs...),)
    end

    @inline function findinds(f_meta, s::Selector)

        # Get all the idcs of the vns in `space` and that belong to the selector `s`
        return filter((i) ->
            (s in f_meta.gids[i] || isempty(f_meta.gids[i])), 1:length(f_meta.gids))
    end

    @inline function findinds(f_meta)
        # Get all the idcs of the vns
        return filter((i) -> isempty(f_meta.gids[i]), 1:length(f_meta.gids))
    end

    """
    `set_retained_vns_del_by_spl!(vi::VarInfo, spl::Sampler)`

    Sets the `"del"` flag of variables in `vi` with `order > vi.num_produce` to `true`.
    """
    function set_retained_vns_del_by_spl!(vi::AbstractVarInfo)
        return set_retained_vns_del_by_spl!(vi, BASE_SELECTOR)
    end

    function set_retained_vns_del_by_spl!(vi::UntypedVarInfo, sel::Selector)
        # Get the indices of `vns` that belong to `spl` as a vector
        gidcs = _getidcs(vi, sel)
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

    function set_retained_vns_del_by_spl!(vi::TypedVarInfo, sel::Selector)
        # Get the indices of `vns` that belong to `spl` as a NamedTuple, one entry for each symbol
        gidcs = _getidcs(vi, sel)
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

    export  VarInfo,
            UntypedVarInfo,
            TypedVarInfo,
            tonamedtuple,
            report_observation!,
            report_transition!,
            set_retained_vns_del_by_spl,
            update_var!,
            initialize
end
