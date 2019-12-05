

abstract type AbstractSMCUtilitFunctions <: AbstractPFUtilitFunctions end
abstract type AbstractPGASUtilityFunctions <: AbstractSMCUtilitFunctions end


struct SMCUtilityFunctions<:AbstractSMCUtilitFunctions
    copy                         ::  Function
    set_retained_vns_del_by_spl! ::  Function
    empty!                       ::  Function
    to_named_tuple               ::  Function
end

const PGUtilityFunctions = SMCUtilityFunctions

struct PGASUtilityFunctions{AP}<:AbstractPGASUtilityFunctions
    copy                         ::  Function
    set_retained_vns_del_by_spl! ::  Function
    merge_traj                   ::  Function
    empty!                       ::  Function
    to_named_tuple               ::  Functino

end
