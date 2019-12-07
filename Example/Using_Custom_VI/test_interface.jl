## Some function which make the model easier to define.


# A very light weight container for a state space model
# The state space is one dimensional
# Note that this is only for a very simple demonstration.



# This is a very shallow container solely for testing puropose
mutable struct Container
    x::Array{Float64,2}
    marked::Vector{Bool}
    produced_at::Vector{Int64}
    num_produce::Float64
end

function Base.deepcopy(vi::Container)
    return Container(deepcopy(vi.x),deepcopy(vi.marked),deepcopy(vi.produced_at),deepcopy(vi.num_produce))
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
    produce(logp)
    trace = current_trace()
end

# logγ corresponds to the proposal distributoin we are sampling from.
function report_transition!(trace,logp::Float64,logγ::Float64)
    trace.taskinfo.logp += logp - logγ
    trace.taskinfo.logpseq += logp
end

function update_var(trace, vn::Int64, r::Vector{Float64})
    if !trace.vi.marked[vn]
        trace.vi.x[vn,:] = r
        trace.vi.marked[vn] = true
        trace.vi.produced_at[vn] = trace.vi.num_produce
        return r
    end
    return trace.vi.x[vn,:]
end


# The reason for this is that we need to pass it!
function copy_container(vi::Container)
    Container(deepcopy(vi.x),deepcopy(vi.marked),deepcopy(vi.produced_at),copy(vi.num_produce))
end

function create_task(f::Function, args...)
    return CTask(() ->  begin new_vi=f(args...); produce(Val{:done}); new_vi; end )
end

function tonamedtuple(vi::Container)
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
vectorize(d::UnivariateDistribution, r::Real) = [r]
vectorize(d::MultivariateDistribution, r::AbstractVector{<:Real}) = copy(r)
vectorize(d::MatrixDistribution, r::AbstractMatrix{<:Real}) = copy(vec(r))
