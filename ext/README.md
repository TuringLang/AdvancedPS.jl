### Guidelines

Extensions allow you to take advantage of any structure your models might have.

API:
- `AdvancedPS.Trace`: Trace container for your custom model type
- `AdvancedPS.advance!`: Emits logdensity
- `AdvancedPS.fork`: Fork particle and create independent child
- `AdvancedPS.forkr`: Fork particle while keeping generated randomness, used for the reference particle in particle MCMC
- `AdvancedPS.update_ref!`: Update reference particle
