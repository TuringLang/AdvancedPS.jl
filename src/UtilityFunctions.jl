
struct SMCUtilityFunctions<:AbstractSMCUtilitFunctions
    copy                         ::  Function
    set_retained_vns_del_by_spl  ::  Function
end

const PGUtilityFunctions = SMCUtilityFunctions

struct PGASUtilityFunctions{AP}<:AbstractPGASUtilityFunctions where AP<:Union{Function,Nothing}
    copy                         ::  Function
    set_retained_vns_del_by_spl  ::  Function
    merge_traj                   ::  Function
    ancestor_proposal            ::  AP
end
