


struct SMCUtilityFunctions<:AbstractSMCUtilitFunctions
    set_retained_vns_del_by_spl! ::  Function
    tonamedtuple                 ::  Function
end



const PGUtilityFunctions = SMCUtilityFunctions

struct PGASUtilityFunctions{AP}<:AbstractPGASUtilityFunctions
    set_retained_vns_del_by_spl! ::  Function
    tonamedtuple                 ::  Function
    merge_traj                   ::  Function
end
