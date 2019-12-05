
###

### The model

###


struct PFModel <: AbstractPFModel
    task::Task
end

function PFModel(f::Function, args)
    task = create_task(f,args)
    PFModel(task)
end


function create_task(f::Function, args=nothing)
    if args === nothing
        return CTask(() ->  begin new_vi=f(); produce(Val{:done}); new_vi; end )
    else
        return CTask(() ->  begin new_vi=f(args...); produce(Val{:done}); new_vi; end )
    end
end
