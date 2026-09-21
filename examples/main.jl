using PythonRNGs
using Random: Random

example1(seed) = rand(PythonRandom(seed), 2, 3)
example2(seed) = rand(NumPyRandom(seed), 2, 3)
example3(rng::Random.AbstractRNG) = rand(rng, 2, 3)

show_matrix(A) = for i in axes(A, 1)
    println("[", join((A[i, j] for j in axes(A, 2)), ", "), "]")
end

println("example1(1234) =")
show_matrix(example1(1234))

println("example2(999) =")
show_matrix(example2(999))

rng = NumPyRandomDefaultRNG(42)
println("example3(rng) =")
show_matrix(example3(rng))
