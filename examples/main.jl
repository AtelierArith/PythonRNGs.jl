using PythonRNGs
using Random: Random

example1(rng::Random.AbstractRNG) = 2rand(rng) + 1
example2(rng::Random.AbstractRNG) = 2rand(rng) + 1
example3(rng::Random.AbstractRNG) = 2rand(rng) + 1

rng = PythonRandom()
Random.seed!(rng, 1234)
@show example1(rng)

rng = NumPyRandomState()
Random.seed!(rng, 999)
@show example2(rng)

rng = NumPyRandomDefaultRNG(42)
@show example3(rng)
