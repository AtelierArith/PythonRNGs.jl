"""
    AbstractPythonRNG <: Random.AbstractRNG

Supertype of random number generators that delegate uniform sampling to a
Python object through [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl).

Concrete subtypes:

- [`PythonRandom`](@ref): backed by Python's standard `random.Random`.
- [`NumPyRandom`](@ref): backed by `numpy.random.default_rng`.
"""
abstract type AbstractPythonRNG <: Random.AbstractRNG end

"""
    NotSupportedError(msg)

Thrown by [`PythonRandom`](@ref) or [`NumPyRandom`](@ref) when a requested draw
has no equivalent in the wrapped Python backend.
"""
struct NotSupportedError <: Exception
    msg::String
end

Base.showerror(io::IO, e::NotSupportedError) = print(io, "Not supported: ", e.msg)

# Both concrete types expose the same fields so that the generic `rand` methods
# and `Random.seed!` can be written against `AbstractPythonRNG`:
#
#   pyobj   : the underlying Python RNG object
#   random  : the bound Python method that returns a uniform float in [0, 1)
#   factory : a Python callable mapping a seed to a fresh RNG object
mutable struct PythonRandom <: AbstractPythonRNG
    pyobj::Py
    random::Py
    factory::Py
end

mutable struct NumPyRandom <: AbstractPythonRNG
    pyobj::Py
    random::Py
    factory::Py
end

"""
    PythonRandom([seed])

Create an `AbstractRNG` that draws uniform random numbers from Python's
standard [`random.Random`](https://docs.python.org/3/library/random.html).

`rand(rng, Float64)` returns exactly the value that `random.Random(seed).random()`
would return on the same draw. When `seed` is omitted the generator is seeded
from the operating system entropy.
"""
function PythonRandom(seed::Union{Integer,Nothing} = nothing)
    factory = pyimport("random").Random
    pyobj = seed === nothing ? factory() : factory(seed)
    return PythonRandom(pyobj, pyobj.random, factory)
end

"""
    NumPyRandom([seed])

Create an `AbstractRNG` that draws uniform random numbers from NumPy's
[`numpy.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html).

`rand(rng, Float64)` returns exactly the value that `default_rng(seed).random()`
would return on the same draw. When `seed` is omitted the generator is seeded
from the operating system entropy.
"""
function NumPyRandom(seed::Union{Integer,Nothing} = nothing)
    np = _pynumpy()
    factory = np.random.default_rng
    pyobj = seed === nothing ? factory() : factory(seed)
    return NumPyRandom(pyobj, pyobj.random, factory)
end

function _pynumpy()
    try
        return pyimport("numpy")
    catch
        throw(
            ArgumentError(
                "NumPyRandom requires NumPy in the Python environment used by " *
                "PythonCall. Install it, for example, with " *
                "`using CondaPkg; CondaPkg.add(\"numpy\")` before loading PythonRNGs.",
            ),
        )
    end
end

function Random.seed!(rng::AbstractPythonRNG, seed::Integer)
    rng.pyobj = rng.factory(seed)
    rng.random = rng.pyobj.random
    return rng
end

function Random.seed!(rng::AbstractPythonRNG)
    rng.pyobj = rng.factory()
    rng.random = rng.pyobj.random
    return rng
end

Random.seed!(rng::AbstractPythonRNG, ::Nothing) = Random.seed!(rng)

"""
    copy(rng::AbstractPythonRNG)

Return an independent copy of `rng` that starts from the same state, using
Python's `copy.deepcopy` on the wrapped generator.
"""
function Base.copy(rng::AbstractPythonRNG)
    pyobj = pyimport("copy").deepcopy(rng.pyobj)
    return typeof(rng)(pyobj, pyobj.random, rng.factory)
end

Base.deepcopy(rng::AbstractPythonRNG) = copy(rng)

Base.show(io::IO, rng::AbstractPythonRNG) = print(io, nameof(typeof(rng)))
