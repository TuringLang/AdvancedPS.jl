

# The idea behind the TaskInfo struct is that it
# allows to propagate information trough the computation.
# This is important because we do not want to

struct TaskInfo <:AbstractTaskInfo
    getidcs::Function # Function to get gicds from sampler
    logpseq::Float64
end
