


struct SMCUtilityFunctions<:AbstractSMCUtilitFunctions
    copy                         ::  Function
    set_retained_vns_del_by_spl! ::  Function
    empty!                       ::  Function
    tonamedtuple                 ::  Function
end



const PGUtilityFunctions = SMCUtilityFunctions

struct PGASUtilityFunctions{AP}<:AbstractPGASUtilityFunctions
    copy                         ::  Function
    set_retained_vns_del_by_spl! ::  Function
    merge_traj                   ::  Function
    empty!                       ::  Function
    tonamedtuple                 ::  Function

end
