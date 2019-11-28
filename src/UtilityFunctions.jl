# Some structure
abstract type AbstractSMCUtilitFunctions <: AbstractPFUtilitFunctions end
abstract type AbstractPGASUtilityFunctions <: AbstractSMCUtilitFunctions end

struct SMCUtilityFunctions<:AbstractSMCUtilitFunctions
    copy                         ::  Function
    set_retained_vns_del_by_spl  ::  Function
end

const PGUtilityFunctions = SMCUtilityFunctions

struct PGASUtilityFunctions{Pr}<:AbstractPGASUtilityFunctions where Pr <:Union{Function,Nothing}
    copy                         ::  Function
    set_retained_vns_del_by_spl  ::  Function
    merge_traj                   ::  Function
    ancestor_proposal            ::  PR
end
