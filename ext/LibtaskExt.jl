module LibtaskExt

import Libtask

isdefined(Base, :get_extension) ? (import AdvancedPS) : (import ..AdvancedPS)

function AdvancedPS.advance!(arg)
    println(arg)
end

end
