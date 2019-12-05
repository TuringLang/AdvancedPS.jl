
v = VarInfo()

vn = @varname hi

push!(v,vn,0.8,Normal())

v[vn] = [1.2]

v2 = VarInfo(v)

typeof(v) <: UntypedVarInfo
typeof(v2) <:UntypedVarInfo

v2.num_produce = 0

fl =  Vector{Float64}(undef,1)
fl[1] = 4.0
setindex!(v2,fl,vn)
v2

v2 = VarInfo{<:NamedTuple}(v)

v2 = TypedVarInfo(v)

v2[vn] = [3.0]

empty!(v2)
