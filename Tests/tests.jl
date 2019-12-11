using Random
using Test
using Distributions
using AdvancedPS
using BenchmarkTools
using Libtask
const APS = AdvancedPS
using Turing
using Turing.Core: @varname

dir = splitdir(splitdir(pathof(AdvancedPS))[1])[1]
push!(LOAD_PATH,dir*"/Example/Using_Turing_VI/" )
using AdvancedPS_Turing_Container
const APSTCont = AdvancedPS_Turing_Container
import AdvancedPS_Turing_Container: tonamedtuple

include(dir*"/Tests/test_utils/AllUtils.jl")


include(dir*"/Tests/test_resample.jl")
include(dir*"/Tests/test_container.jl")
include(dir*"/Tests/Using_Turing_VI/numerical_tests.jl")
include(dir*"/Tests/Using_Turing_VI/large_numerical_tests.jl")
