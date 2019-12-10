
###

### The model

###


struct PFModel{NT} <: AbstractPFModel where {NT <: Union{Tuple,NamedTuple}}
    task::Task
    args::NT
end


function PFModel(f::Function, args::NamedTuple)
    task = create_task(f, args)
    PFModel(task, args)
end


function create_task(f::Function, args::NamedTuple)
    return CTask(() ->  begin new_vi=f(args...); produce(Val{:done}); new_vi; end )
end
