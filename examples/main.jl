using PythonRNGs
using Random: Random

example1(seed) = 2rand(PythonRandom(seed)) + 1
example2(seed) = 2rand(NumPyRandom(seed)) + 1
example3(rng::Random.AbstractRNG) = 2rand(rng) + 1

@show example1(1234)
@show example2(999)

rng = NumPyRandomDefaultRNG(42)
@show example3(rng)
