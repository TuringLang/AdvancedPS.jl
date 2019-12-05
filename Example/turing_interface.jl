## Some function which make the model easier to define.




# This is important for initalizaiton
const initialize = AdvancedPS.current_trace

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



function in(sym::Symbol, vns::Vector{VarName})::Bool

end


# We obtain the new value for our variable
# If the distribution is not specified, we simply set it to be Normal
function update_var(trace, vn, val, dist= Normal())

    # check if the symbol is contained in the varinfo...
    if haskey(trace.vi,vn)
        if is_flagged(trace.vi, vn, "del")
            unset_flag!(vi, vn, "del")
            trace.vi[vn] = val
            setgid!(vi, spl.selector, vn)
            setorder!(vi, vn, vi.num_produce)
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
