## Some function which make the model easier to define.


# A very light weight container for a state space model
# The state space is one dimensional
# Note that this is only for a very simple demonstration.

module AdvancedPS_SSM_Container
    using Libtask
    using AdvancedPS: current_trace
    using NamedTupleTools
    # This is a very shallow container solely for testing puropose
    abstract type AbstractPFContainer end
    export AbstractPFContainer

    const initialize = current_trace

    mutable struct Container{M<:AbstractMatrix, I<:Integer} <: AbstractPFContainer
        x::M
        marked::Vector{Bool}
        produced_at::Vector{I}
        num_produce::I
    end



    function Base.copy(vi::Container)
        return Container(deepcopy(vi.x),deepcopy(vi.marked),deepcopy(vi.produced_at),deepcopy(vi.num_produce))
    end


    @inline function set_retained_vns_del_by_spl!(container::Container)
        for i in 1:length(container.marked)
            if container.marked[i]
                if container.produced_at[i] > container.num_produce
                    container.marked[i] = false
                end
            end
        end
        container
    end



    # This is important for initalizaiton

    @inline function report_observation!(trace, logp::Real)
        produce(logp)
        current_trace()
    end

    # logγ corresponds to the proposal distributoin we are sampling from.
    @inline function report_transition!(trace ,logp::Real,logγ::Real)
        trace.taskinfo.logp += logp - logγ
        trace.taskinfo.logpseq += logp
    end

    @inline function update_var!(trace, vn::Int, r::Vector{<:Real})
        if !trace.vi.marked[vn]
            trace.vi.x[vn,:] = r # We assume that it is already vectorized!!
            trace.vi.marked[vn] = true
            trace.vi.produced_at[vn] = trace.vi.num_produce
            return r
        end
        return trace.vi.x[vn,:]
    end



    @inline function is_marked(trace, vn::Int)::Bool
        if trace.vi.marked[vn]
            return true
        end
        return false
    end

    @inline function get_vn(trace, vn::Int)
        return trace.vi.x[vn,:]
    end

    @inline function get_traj(trace)
        return trace.vi.x[1:trace.vi.num_produce+1,:]
    end

    # The reason for this is that we need to pass it!
    @inline function Base.copy(vi::Container)
        Container(deepcopy(vi.x),deepcopy(vi.marked),deepcopy(vi.produced_at),copy(vi.num_produce))
    end

    @inline function tonamedtuple(vi::Container)
        tnames = Tuple([Symbol("x$i") for i in 1:size(vi.x)[1]])
        tvalues = Tuple([vi.x[i,:] for i in 1:size(vi.x)[1]])
        return namedtuple(tnames, tvalues)
    end

    function Base.empty!(vi::Container)
        for i in 1:length(vi.marked)
            vi.marked[i] = false
        end
        vi
    end

    export  Container,
            tonamedtuple,
            get_traj,
            get_vn,
            is_marked,
            report_observation!,
            report_transition!,
            set_retained_vns_del_by_spl,
            update_var!,
            initialize


end
