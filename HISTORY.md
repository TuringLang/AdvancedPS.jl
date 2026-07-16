# 0.8.0

## Breaking changes

Dropped the `Requires` dependency.
The Libtask extension now loads only through native package extensions, so Julia 1.10 or later is required (already the minimum).

Raised compat lower bounds: `AbstractMCMC` to 5, `Distributions` to 0.25, and `StatsFuns` to 1 (now also allowing 2).
