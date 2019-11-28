



## Some function which make the model easier to define.











# A very light weight container for a state space model



# The state space is one dimensional!



# Note that this is only for a very simple demonstration.



mutable struct Container



    x::Vector{Float64}



    num_produce::Float64



end







# This is important for initalizaiton



function initialize()



    return current_trace()



end







# Th



function report_observation(trace, logp::Float64)



    trace.taskinfo.logp += logp



    produce(logp)



    current_trace()



end







# logγ corresponds to the proposal distributoin we are sampling from.



function report_transition(trace,logp::Float64,logγ::Float64)



    trace.taskinfo.logp += logp - logγ



    trace.taskinfo.logpseq += logp



end







function get_x(trace,indx)



    return @inbounds trace.vi.x[indx]



end



function set_x(trace,indx,val)



    return @inbounds trace.vi.x[indx] = val



end











function copyC(vi::Container)



    Container(deepcopy(vi.x),vi.num_produce)



end







function create_task(f::Function)



    return CTask(() ->  begin new_vi=f(); produce(Val{:done}); new_vi; end )



end
