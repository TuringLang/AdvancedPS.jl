# API

## Samplers

AdvancedPS introduces a few samplers extending [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl).
The `sample` method expects a custom type that subtypes `AbstractMCMC.AbstractModel`.
The available samplers are listed below:

### SMC

```@docs
AdvancedPS.SMC
```
The SMC sampler populates a set of particles in a [`AdvancedPS.ParticleContainer`](@ref) and performs a [`AdvancedPS.sweep!`](@ref) which 
propagates the particles and provides an estimation of the log-evidence

```julia
sampler = SMC(nparticles) 
chains = sample(model, sampler)
```

### Particle Gibbs
```@docs
AdvancedPS.PG
```
The Particle Gibbs introduced in [^2] runs a sequence of conditional SMC steps where a pre-selected particle, the reference particle, is replayed and propagated through 
the SMC step.

```julia
sampler = PG(nparticles)
chains = sample(model, sampler, nchains)
```

For more detailed examples please refer to the [Examples](@ref) page.

## Resampling

AdvancedPS implements adaptive resampling for both [`AdvancedPS.PG`](@ref) and [`AdvancedPS.SMC`](@ref).
The following resampling schemes are implemented:
```@docs
AdvancedPS.resample_multinomial
AdvancedPS.resample_residual
AdvancedPS.resample_stratified
AdvancedPS.resample_systematic
```

Each of these schemes is wrapped in a [`AdvancedPS.ResampleWithESSThreshold`](@ref) struct to trigger a resampling step whenever the ESS is below a certain threshold.
```@docs
AdvancedPS.ResampleWithESSThreshold
```

## RNG

AdvancedPS replays the individual trajectories instead of storing the intermediate values. This way we can build efficient samplers. 
However in order to replay the trajectories we need to reproduce most of the random numbers generated 
during the execution of the program while also generating diverging traces after each resampling step. 
To solve these two issues AdvancedPS uses counter-based RNG introduced in [^1] and widely used in large parallel systems see 
[StochasticDifferentialEquations](https://github.com/SciML/StochasticDiffEq.jl) or [JAX](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html?highlight=random)
for other implementations. 

Under the hood AdvancedPS is using [Random123](https://github.com/JuliaRandom/Random123.jl) for the generators.
Using counter-based RNG allows us to split generators thus creating new independent random streams. These generators are also wrapped in a [`AdvancedPS.TracedRNG`](@ref) type. 
The `TracedRNG` keeps track of the keys generated at every `split` and can be reset to replay random streams.


```@docs
AdvancedPS.TracedRNG
AdvancedPS.split
AdvancedPS.load_state!
AdvancedPS.save_state!
```

## Internals
### Particle Sweep

```@docs
AdvancedPS.ParticleContainer
AdvancedPS.sweep!
```

[^1]: John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. 2011. Parallel random numbers: as easy as 1, 2, 3. In Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis (SC '11). Association for Computing Machinery, New York, NY, USA, Article 16, 1–12. DOI:https://doi.org/10.1145/2063384.2063405
[^2]: Andrieu, Christophe, Arnaud Doucet, and Roman Holenstein. "Particle Markov chain Monte Carlo methods." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 72, no. 3 (2010): 269-342.
