# Retrieve name of example and output directory
if length(ARGS) != 2
    error("please specify the name of the example and the output directory")
end
const EXAMPLE = ARGS[1]
const OUTDIR = ARGS[2]

# Activate environment
# Note that each example's Project.toml must include Literate as a dependency
using Pkg: Pkg
const EXAMPLEPATH = joinpath(@__DIR__, "..", "examples", EXAMPLE)
Pkg.activate(EXAMPLEPATH)
Pkg.instantiate()
using Literate: Literate

# Convert to markdown and notebook
const SCRIPTJL = joinpath(EXAMPLEPATH, "script.jl")
Literate.markdown(SCRIPTJL, OUTDIR; name=EXAMPLE, execute=true)
