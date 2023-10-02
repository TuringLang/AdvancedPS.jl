abstract type AbstractReferenceSampler end

struct IdentityReferenceSampler <: AbstractReferenceSampler end # PG
struct AncestorReferenceSampler <: AbstractReferenceSampler end # PG-AS
