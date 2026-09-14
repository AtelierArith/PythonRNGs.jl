module PythonRNGs

using Random
using PythonCall

export AbstractPythonRNG,
    PythonRandom, NumPyRandom, NumPyRandomDefaultRNG, NotSupportedError

include("backends.jl")
include("rand.jl")
include("normal.jl")
include("permutations.jl")

end # module PythonRNGs
