# Uniform sampling for Python-backed RNGs.
#
# `rand(rng, Float64)` is drawn directly from the Python generator, and integer
# draws delegate to each backend's own integer API, so those values match what
# Python itself produces. Float32/Float16, which Python's standard library does
# not provide natively, are derived from a native 32-bit integer draw.

# Route `rand(rng, Float64)` (and its Float16/Float32 relatives) through
# `SamplerType` instead of the default `CloseOpen01` machinery.
Random.Sampler(
    ::Type{<:AbstractPythonRNG},
    ::Type{T},
    ::Random.Repetition,
) where {T<:Union{Float16,Float32,Float64}} = Random.SamplerType{T}()

# Tell the default samplers (e.g. ranges) to build 52-bit draws from `UInt64`.
Random.rng_native_52(::AbstractPythonRNG) = UInt64

@inline _rand_float64(rng::AbstractPythonRNG) = pyconvert(Float64, rng.random())

# Array draws follow Python's C order (last index varies fastest), including
# writes into existing arrays and views.
Random.rand(rng::AbstractPythonRNG, ::Type{T}, dims::Dims) where {T} =
    _rand_array(rng, T, dims)
Random.rand(rng::AbstractPythonRNG, X, dims::Dims) = _rand_array(rng, X, dims)

function _rand_array(rng::AbstractPythonRNG, X, dims::Dims)
    A = Array{Random.gentype(X)}(undef, dims)
    return rand!(rng, A, Random.Sampler(rng, X))
end

Random.rand!(rng::AbstractPythonRNG, A::AbstractArray, sp::Random.Sampler) =
    _rand_c_order!(rng, A, sp)

# Resolve Random's specialized BitArray method while using the same draw order.
Random.rand!(rng::AbstractPythonRNG, A::BitArray, sp::Random.SamplerType{Bool}) =
    _rand_c_order!(rng, A, sp)

function _rand_c_order!(rng::AbstractPythonRNG, A::AbstractArray, sp::Random.Sampler)
    for I in CartesianIndices(reverse(axes(A)))
        @inbounds A[reverse(Tuple(I))...] = rand(rng, sp)
    end
    return A
end

function Random.rand!(
    rng::Union{NumPyRandomDefaultRNG,NumPyRandom}, A::AbstractArray,
    ::Random.SamplerType{Float64},
)
    return copyto!(A, _rand_array(rng, Float64, size(A)))
end

function Random.rand!(
    rng::NumPyRandomDefaultRNG, A::AbstractArray, ::Random.SamplerType{Float32},
)
    return copyto!(A, _rand_array(rng, Float32, size(A)))
end

# Converting a NumPy array preserves logical indices, regardless of the
# different memory layouts. Native array calls also avoid per-element calls.
function _rand_array(
    rng::Union{NumPyRandomDefaultRNG,NumPyRandom}, ::Type{Float64}, dims::Dims,
)
    all(d -> d >= 0, dims) || throw(ArgumentError("array dimensions must be non-negative"))
    return pyconvert(Array{Float64,length(dims)}, rng.random(size = dims))
end

function _rand_array(rng::NumPyRandomDefaultRNG, ::Type{Float32}, dims::Dims)
    all(d -> d >= 0, dims) || throw(ArgumentError("array dimensions must be non-negative"))
    return pyconvert(
        Array{Float32,length(dims)}, rng.pyobj.random(size = dims, dtype = "float32"),
    )
end

# Native uniform integers.
#
# Python's `random.Random` exposes `getrandbits(k)`; NumPy's `Generator`
# exposes `integers` (bounded) and `BitGenerator.random_raw` (full 64 bits);
# the legacy `RandomState` exposes `randint` (bounded). Delegating to these
# keeps `rand(rng, T)` equal to the corresponding Python draw for the same seed.

_rand_uint(rng::PythonRandom, ::Type{UInt8}) = pyconvert(UInt8, rng.pyobj.getrandbits(8))
_rand_uint(rng::PythonRandom, ::Type{UInt16}) = pyconvert(UInt16, rng.pyobj.getrandbits(16))
_rand_uint(rng::PythonRandom, ::Type{UInt32}) = pyconvert(UInt32, rng.pyobj.getrandbits(32))
_rand_uint(rng::PythonRandom, ::Type{UInt64}) = pyconvert(UInt64, rng.pyobj.getrandbits(64))
_rand_uint(rng::PythonRandom, ::Type{UInt128}) =
    pyconvert(UInt128, rng.pyobj.getrandbits(128))
_rand_bool(rng::PythonRandom) = pyconvert(Bool, rng.pyobj.getrandbits(1))

_rand_uint(rng::NumPyRandomDefaultRNG, ::Type{UInt8}) =
    pyconvert(UInt8, rng.pyobj.integers(0, 0x0100, dtype = "uint8"))
_rand_uint(rng::NumPyRandomDefaultRNG, ::Type{UInt16}) =
    pyconvert(UInt16, rng.pyobj.integers(0, 0x10000, dtype = "uint16"))
_rand_uint(rng::NumPyRandomDefaultRNG, ::Type{UInt32}) =
    pyconvert(UInt32, rng.pyobj.integers(0, 0x100000000, dtype = "uint32"))
_rand_uint(rng::NumPyRandomDefaultRNG, ::Type{UInt64}) =
    pyconvert(UInt64, rng.pyobj.bit_generator.random_raw())
_rand_uint(rng::NumPyRandomDefaultRNG, ::Type{UInt128}) =
    (UInt128(_rand_uint(rng, UInt64)) << 64) | UInt128(_rand_uint(rng, UInt64))
_rand_bool(rng::NumPyRandomDefaultRNG) =
    pyconvert(UInt8, rng.pyobj.integers(0, 0x02, dtype = "uint8")) == 0x01

# The legacy `RandomState` exposes `randint` (high exclusive) instead of
# `integers`; `high = 2^N` is passed as a Python arbitrary-precision int so that
# the full `UInt64` range works.
_rand_uint(rng::NumPyRandom, ::Type{UInt8}) =
    pyconvert(UInt8, rng.pyobj.randint(0, 0x0100, dtype = "uint8"))
_rand_uint(rng::NumPyRandom, ::Type{UInt16}) =
    pyconvert(UInt16, rng.pyobj.randint(0, 0x10000, dtype = "uint16"))
_rand_uint(rng::NumPyRandom, ::Type{UInt32}) =
    pyconvert(UInt32, rng.pyobj.randint(0, 0x100000000, dtype = "uint32"))
_rand_uint(rng::NumPyRandom, ::Type{UInt64}) =
    pyconvert(UInt64, rng.pyobj.randint(0, big(2)^64, dtype = "uint64"))
_rand_uint(rng::NumPyRandom, ::Type{UInt128}) =
    (UInt128(_rand_uint(rng, UInt64)) << 64) | UInt128(_rand_uint(rng, UInt64))
_rand_bool(rng::NumPyRandom) =
    pyconvert(UInt8, rng.pyobj.randint(0, 0x02, dtype = "uint8")) == 0x01

# NumPy's `Generator.random` has a native `Float32`, so match it exactly.
Random.rand(rng::NumPyRandomDefaultRNG, ::Random.SamplerType{Float32}) =
    pyconvert(Float32, rng.pyobj.random(dtype = "float32"))

# Floats
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Float64}) = _rand_float64(rng)

Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Float32}) =
    reinterpret(Float32, (_rand_uint(rng, UInt32) & 0x007fffff) | 0x3f800000) - 1.0f0

Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Float16}) = Float16(
    reinterpret(Float32, ((_rand_uint(rng, UInt32) & 0x000003ff) << 13) | 0x3f800000) -
    1.0f0,
)

# Unsigned integers
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{UInt8}) = _rand_uint(rng, UInt8)
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{UInt16}) = _rand_uint(rng, UInt16)
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{UInt32}) = _rand_uint(rng, UInt32)
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{UInt64}) = _rand_uint(rng, UInt64)
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{UInt128}) =
    _rand_uint(rng, UInt128)

# Signed integers: reinterpret the native unsigned draw (uniform over the full
# range, matching `reinterpret(T, rand(rng, unsigned(T)))`).
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Int8}) =
    _rand_uint(rng, UInt8) % Int8
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Int16}) =
    _rand_uint(rng, UInt16) % Int16
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Int32}) =
    _rand_uint(rng, UInt32) % Int32
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Int64}) =
    _rand_uint(rng, UInt64) % Int64
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Int128}) =
    _rand_uint(rng, UInt128) % Int128

# Boolean
Random.rand(rng::AbstractPythonRNG, ::Random.SamplerType{Bool}) = _rand_bool(rng)

# Integer ranges.
#
# `rand(rng, a:b)` delegates to the backend's native range API so the values
# match Python:
#
#   PythonRandom -> `random.Random.randint(a, b)` for `a:b`, and `randrange`
#                   for step ranges
#   NumPyRandomDefaultRNG -> `Generator.integers(a, b, endpoint = true, dtype = ...)`
#   NumPyRandom      -> `RandomState.randint(a, b + 1, dtype = ...)`
#
# Ranges with no NumPy equivalent throw a `NotSupportedError`.

_numpy_dtype(::Type{Int8}) = "int8"
_numpy_dtype(::Type{Int16}) = "int16"
_numpy_dtype(::Type{Int32}) = "int32"
_numpy_dtype(::Type{Int64}) = "int64"
_numpy_dtype(::Type{UInt8}) = "uint8"
_numpy_dtype(::Type{UInt16}) = "uint16"
_numpy_dtype(::Type{UInt32}) = "uint32"
_numpy_dtype(::Type{UInt64}) = "uint64"

const NumPySupportedInt = Union{Int8,Int16,Int32,Int64,UInt8,UInt16,UInt32,UInt64}
const NumPyUnsupportedInt = Union{Int128,UInt128,BigInt}

struct PythonRangeSampler{T,R<:AbstractRange} <: Random.Sampler{T}
    r::R
end

struct NumPyRangeSampler{T,R<:AbstractRange} <: Random.Sampler{T}
    r::R
end

function _range_length(r::AbstractRange)
    n = try
        length(r)
    catch
        throw(NotSupportedError("cannot compute the length of a $(typeof(r))"))
    end
    n >= 1 || throw(ArgumentError("collection must be non-empty"))
    n <= typemax(Int) || throw(
        NotSupportedError("ranges with more than typemax(Int) elements are not supported"),
    )
    return Int(n)
end

Random.Sampler(rng::PythonRandom, r::AbstractRange, ::Random.Repetition) =
    PythonRangeSampler{eltype(r),typeof(r)}(r)

Random.Sampler(rng::NumPyRandomDefaultRNG, r::AbstractRange, ::Random.Repetition) =
    NumPyRangeSampler{eltype(r),typeof(r)}(r)

Random.Sampler(rng::NumPyRandom, r::AbstractRange, ::Random.Repetition) =
    NumPyRangeSampler{eltype(r),typeof(r)}(r)

function Random.rand(rng::PythonRandom, sp::PythonRangeSampler)
    r = sp.r
    if eltype(r) <: Integer && r isa AbstractUnitRange
        return pyconvert(eltype(r), rng.pyobj.randint(first(r), last(r)))
    else
        k = pyconvert(Int, rng.pyobj.randrange(_range_length(r)))
        return @inbounds r[begin+k]
    end
end

function Random.rand(rng::NumPyRandomDefaultRNG, sp::NumPyRangeSampler{T}) where {T}
    r = sp.r
    if T <: NumPyUnsupportedInt
        throw(
            NotSupportedError(
                "NumPyRandomDefaultRNG does not support ranges of eltype $T; supported integer types " *
                "are Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64",
            ),
        )
    elseif T <: NumPySupportedInt && r isa AbstractUnitRange
        v = rng.pyobj.integers(first(r), last(r), endpoint = true, dtype = _numpy_dtype(T))
        return pyconvert(T, v)
    else
        k = pyconvert(Int, rng.pyobj.integers(0, _range_length(r), dtype = "int64"))
        return @inbounds r[begin+k]
    end
end

function Random.rand(rng::NumPyRandom, sp::NumPyRangeSampler{T}) where {T}
    r = sp.r
    if T <: NumPyUnsupportedInt
        throw(
            NotSupportedError(
                "NumPyRandom does not support ranges of eltype $T; supported integer " *
                "types are Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64",
            ),
        )
    elseif T <: NumPySupportedInt && r isa AbstractUnitRange
        # `randint` takes an exclusive upper bound; use Python ints to avoid
        # overflow for full ranges such as `typemax(UInt64)`.
        v = rng.pyobj.randint(big(first(r)), big(last(r)) + 1, dtype = _numpy_dtype(T))
        return pyconvert(T, v)
    else
        k = pyconvert(Int, rng.pyobj.randint(0, _range_length(r), dtype = "int64"))
        return @inbounds r[begin+k]
    end
end
