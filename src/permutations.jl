# Python permutations are zero-based; Julia's randperm returns 1:n.
function Random.randperm(rng::AbstractPythonRNG, n::T) where {T<:Integer}
    n >= 0 || throw(ArgumentError("permutation length must be non-negative"))
    count = Int(n)
    values = _backend_permutation(rng, count)
    return T[x + 1 for x in values]
end

_backend_permutation(rng::PythonRandom, n::Int) =
    pyconvert(Vector{Int}, rng.pyobj.sample(pybuiltins.range(n), n))
_backend_permutation(rng::Union{NumPyRandomDefaultRNG,NumPyRandom}, n::Int) =
    pyconvert(Vector{Int}, rng.pyobj.permutation(n))

# Shuffle indices in Python, then move the original Julia values. This retains
# object identity and supports element types with no NumPy dtype.
function _backend_shuffle(rng::PythonRandom, n::Int)
    indices = pybuiltins.list(pybuiltins.range(n))
    rng.pyobj.shuffle(indices)
    return pyconvert(Vector{Int}, indices)
end
function _backend_shuffle(rng::Union{NumPyRandomDefaultRNG,NumPyRandom}, n::Int)
    indices = _pynumpy().arange(n)
    rng.pyobj.shuffle(indices)
    return pyconvert(Vector{Int}, indices)
end

function _shuffle_vector!(rng::AbstractPythonRNG, A::AbstractVector)
    original = collect(A)
    order = _backend_shuffle(rng, length(A))
    for (i, k) in zip(eachindex(A), order)
        A[i] = original[k + 1]
    end
    return A
end

Random.shuffle!(rng::AbstractPythonRNG, A::AbstractVector) = _shuffle_vector!(rng, A)
# Disambiguate Random's Boolean-array specialization on newer Julia versions.
Random.shuffle!(rng::AbstractPythonRNG, A::AbstractVector{Bool}) = _shuffle_vector!(rng, A)
Random.shuffle(rng::AbstractPythonRNG, A::AbstractVector) = shuffle!(rng, collect(A))
# Random otherwise routes OneTo through randperm, which differs from Python sample.
Random.shuffle(rng::AbstractPythonRNG, A::Base.OneTo) = shuffle!(rng, collect(A))
