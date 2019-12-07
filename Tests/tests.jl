using Random
using Test
using Distributions
using Turing # Compiler3.0 branch!!
using AdvancedPS

dir = splitdir(splitdir(pathof(AdvancedPS))[1])[1]

include(dir*"/Example/Using_Turing_VI/turing_interface.jl")
include(dir*"/Tests/test_utils/AllUtils.jl")

import Turing.Core: tonamedtuple
tonamedtuple(vi::UntypedVarInfo) = tonamedtuple(TypedVarInfo(vi))



include(dir*"/Tests/test_resample.jl")
include(dir*"/Tests/test_container.jl")
include(dir*"/Tests/Using_Turing_VI/numerical_tests.jl")
include(dir*"/Tests/Using_Turing_VI/large_numerical_tests.jl")
