# Reproducibility

For the same seed, each draw equals the corresponding Python call:

| Draw | `PythonRandom` | `NumPyRandomDefaultRNG` | `NumPyRandom` |
| --- | --- | --- | --- |
| `rand(rng, Float64)` | `random.Random(seed).random()` | `default_rng(seed).random()` | `RandomState(seed).random_sample()` |
| `rand(rng, n)` (array) | `[random.Random(seed).random() for _ in range(n)]` | `default_rng(seed).random(n)` | `RandomState(seed).random_sample(n)` |
| `rand(rng, UIntN)` | `getrandbits(N)` | `N = 8, 16, 32`: `integers(0, 2^N, dtype="uintN")`; `N = 64`: `bit_generator.random_raw()`; `N = 128`: two 64-bit draws | `N = 8, 16, 32, 64`: `randint(0, 2^N, dtype="uintN")`; `N = 128`: two 64-bit draws |
| `rand(rng, Bool)` | `bool(getrandbits(1))` | `integers(0, 2, dtype="uint8")` | `randint(0, 2, dtype="uint8")` |
| `rand(rng, Float32)` | derived from the native 32-bit integer draw | `random(dtype="float32")` | derived from the native 32-bit integer draw |
| `rand(rng, a:b)` | `randint(a, b)` | `integers(a, b, endpoint=true, dtype=...)` | `randint(a, b + 1, dtype=...)` |
| `rand(rng, a:s:b)` | `randrange(a, b + s, s)` | index from the native integer draw | index from the native integer draw |
| `rand(rng, 'a':'z')`, discrete float range | `choice(collect(r))` | `Generator.choice(collect(r))` | `RandomState.choice(collect(r))` |

## Notes

- `NumPyRandom` corresponds to the global legacy API, so
  `rand(rng::NumPyRandom, Float64)` equals
  `np.random.seed(seed); np.random.random()`.
- Signed integers reinterpret the native unsigned draw, e.g.
  `rand(rng, Int64) == reinterpret(Int64, rand(rng, UInt64))`.
- `rand(rng, a:b)` also equals the corresponding `choice` of the range values:
  `random.choice([a, ..., b])` for `PythonRandom` (because `randint` and `choice`
  both use `_randbelow`), and `Generator.choice` / `RandomState.choice` for the
  NumPy backends.
- `Float16`, `PythonRandom`'s `Float32`, and `NumPyRandom`'s `Float32` are
  derived from a native 32-bit integer draw: Python's standard library has no
  native `Float32`, NumPy's `random` has no `Float16`, and `RandomState` has no
  `Float32`/`Float16`.
- `NumPyRandomDefaultRNG` and `NumPyRandom` throw `NotSupportedError` for integer
  ranges whose element type NumPy has no dtype for (`Int128`, `UInt128`,
  `BigInt`); `PythonRandom` supports those.
- Both `rand(rng, ..., dims...)` and `rand!(rng, A, ...)` fill in C order
  (last index varies fastest), matching Python's logical indices. For the same seed
  and preceding calls, `jlarray[begin+i, begin+j] == nparray[i, j]`.
  Julia arrays still use column-major storage; only the draw assignment changes.
  - `rand(rng::NumPyRandomDefaultRNG, dims...)` equals
    `default_rng(seed).random(size=dims)` elementwise.
  - `rand(rng::NumPyRandom, dims...)` equals
    `RandomState(seed).random_sample(size=dims)` elementwise.
  - `PythonRandom` draws repeatedly from a single `random.Random(seed)` instance
    and fills in C order, equivalent to reshaping the resulting list with
    `numpy.array(values).reshape(dims)`.
  - NumPy `Float64` arrays and `NumPyRandomDefaultRNG` `Float32` arrays are
    generated in one backend call. Other types and collections use repeated
    scalar draws in C order; these need not match NumPy's bulk integer APIs,
    which can consume the random stream differently.
  - This changes the previous column-major assignment for multidimensional
    draws, including `rand!`. For an existing array or view, `rand!(rng, A)`
    assigns the same values as `rand(rng, eltype(A), size(A))` from the same
    RNG state, and returns `A` itself.
- Only uniform distributions are provided. Distributions such as `Normal` or
  `Exponential` are intentionally out of scope; draw a uniform value and
  transform it yourself if needed.

## Array order

Both allocating and in-place draws assign values in C order: the last index
varies fastest. For a 2×3 array, six successive values `a, b, c, d, e, f`
occupy these positions:

```text
a b c
d e f
```

```julia
using PythonRNGs, PythonCall, Random

A = rand(NumPyRandomDefaultRNG(42), 2, 3)
B = zeros(2, 3)
@assert rand!(NumPyRandomDefaultRNG(42), B) === B
@assert A == B

np = pyimport("numpy")
nparray = np.random.default_rng(42).random(size = (2, 3))
for i in 0:1, j in 0:2
    @assert B[begin+i, begin+j] == pyconvert(Float64, nparray[i, j])
end
```

`A` and `B` are Julia `Matrix{Float64}` values, not `Py` wrappers. The same
index correspondence applies in higher dimensions. Type and range arguments,
as well as destinations that are views, also use C-order assignment.
