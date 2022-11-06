# AdvancedPS

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/AdvancedPS.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/AdvancedPS.jl/dev)
[![Build Status](https://github.com/TuringLang/AdvancedPS.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/AdvancedPS.jl/actions?query=workflow%3ACI%20branch%3Amaster)
[![Coverage](https://codecov.io/gh/TuringLang/AdvancedPS.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/AdvancedPS.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

AdvancedPS provides an efficient implementation of common particle based Monte Carlo samplers using the [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl) interface.
The package also relies on [Libtask](https://github.com/TuringLang/Libtask.jl) for task manipulation.
AdvancedPS is part of the [Turing](https://turing.ml/stable/) ecosystem.

### Installation

Inside the Julia REPL
```julia
julia>] add AdvancedPS
```

### Examples

Detailed examples are available in the [documentation](https://turinglang.github.io/AdvancedPS.jl/dev/)

### Reference

1. Doucet, Arnaud, and Adam M. Johansen. "A tutorial on particle filtering and smoothing: Fifteen years later." Handbook of nonlinear filtering 12, no. 656-704 (2009): 3.

2. Andrieu, Christophe, Arnaud Doucet, and Roman Holenstein. "Particle Markov chain Monte Carlo methods." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 72, no. 3 (2010): 269-342.

3. Tripuraneni, Nilesh, Shixiang Shane Gu, Hong Ge, and Zoubin Ghahramani. "Particle gibbs for infinite hidden Markov models." In Advances in Neural Information Processing Systems, pp. 2395-2403. 2015.

4. Lindsten, Fredrik, Michael I. Jordan, and Thomas B. Schön. "Particle Gibbs with ancestor sampling." The Journal of Machine Learning Research 15, no. 1 (2014): 2145-2184.

5. Pitt, Michael K., and Neil Shephard. "Filtering via simulation: Auxiliary particle filters." Journal of the American statistical association 94, no. 446 (1999): 590-599.

6. Doucet, Arnaud, Nando de Freitas, and Neil Gordon. "Sequential Monte Carlo Methods in Practice."

7. Del Moral, Pierre, Arnaud Doucet, and Ajay Jasra. "Sequential Monte Carlo samplers." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 68, no. 3 (2006): 411-436.

