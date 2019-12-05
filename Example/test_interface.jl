## Some function which make the model easier to define.


# A very light weight container for a state space model
# The state space is one dimensional
# Note that this is only for a very simple demonstration.



# This is a very shallow container solely for testing puropose
mutable struct Container
    x::Array{Float64,1}
    marked::Vector{Bool}
    produced_at::Vector{Int64}
    num_produce::Float64
end

function set_retained_vns_del_by_spl!(container::Container)
    for i in 1:length(container.marked)
        if container.marked[i]
            if container.produced_at[i] >= container.num_produce
                container.marked[i] = false
            end
        end
    end
    container
end



# This is important for initalizaiton
const initialize = current_trace

function report_observation!(trace, logp::Float64)
    trace.taskinfo.logp += logp
    trace.vi.num_produce += 1
    produce(logp)
    trace = current_trace()
end

# logγ corresponds to the proposal distributoin we are sampling from.
function report_transition!(trace,logp::Float64,logγ::Float64)
    trace.taskinfo.logp += logp - logγ
    trace.taskinfo.logpseq += logp
end

function get_x(trace,indx)
    @assert trace.vi.marked[indx] "[Interface] This should already be marked!"
    return trace.vi.x[indx]
end

# We only set it if it is not yet marked
function set_x!(trace,indx,val)
    if !trace.vi.marked[indx]
        trace.vi.x[indx] = val
        trace.vi.marked[indx] = true
        trace.vi.produced_at[indx] = trace.vi.num_produce
    end
    trace
end

# The reason for this is that we need to pass it!
function copy_container(vi::Container)
    Container(deepcopy(vi.x),deepcopy(vi.marked),deepcopy(vi.produced_at),copy(vi.num_produce))
end

function create_task(f::Function)
    return CTask(() ->  begin new_vi=f(); produce(Val{:done}); new_vi; end )
end
