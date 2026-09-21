# PythonRNGs examples

The same three examples written twice:

- [`main.py`](main.py) uses Python's `random` and `numpy` directly.
- [`main.jl`](main.jl) uses `PythonRNGs`, and produces **exactly the same
  numbers** by driving the same Python generators.

These examples are not about speed: `PythonRNGs` calls into Python, so it is
slower than a native Julia RNG. They show how to **validate a Julia port of
Python code** by checking that the translated routine matches the original
Python implementation value-for-value.

Each example draws a 2×3 matrix.

| Example | Python (`main.py`) | Julia (`main.jl`) | Seed |
| --- | --- | --- | --- |
| `example1` | `random.Random(seed)` + `rng.random()` | `PythonRandom` | 1234 |
| `example2` | `np.random.RandomState(seed)` + `rng.random((2, 3))` (legacy) | `NumPyRandom` | 999 |
| `example3` | `np.random.default_rng(seed)` + `rng.random((2, 3))` | `NumPyRandomDefaultRNG` | 42 |

## Expected output

Julia:

```console
$ julia --project=.. main.jl
example1(1234) =
[0.9664535356921388, 0.4407325991753527, 0.007491470058587191]
[0.9109759624491242, 0.939268997363764, 0.5822275730589491]
example2(999) =
[0.8034280400796879, 0.5275222956826447, 0.11911146502821202]
[0.6396814442423551, 0.09092526277407997, 0.3322256807930123]
example3(rng) =
[0.7739560485559633, 0.4388784397520523, 0.8585979199113825]
[0.6973680290593639, 0.09417734788764953, 0.9756223516367559]
```

Python:

```console
$ uv run main.py
example1(1234) =
[[0.9664535356921388, 0.4407325991753527, 0.007491470058587191], [0.9109759624491242, 0.939268997363764, 0.5822275730589491]]
example2(999) =
[[0.8034280400796879  0.5275222956826447  0.11911146502821202]
 [0.6396814442423551  0.09092526277407997 0.3322256807930123 ]]
example3(rng) =
[[0.7739560485559633  0.4388784397520523  0.8585979199113825 ]
 [0.6973680290593639  0.09417734788764953 0.9756223516367559 ]]
```

The values and the draw order (row-major) match for every example.

## Running

### Python

This directory is a [uv](https://docs.astral.sh/uv/) project that depends on
NumPy. uv creates the environment on first run:

```console
$ uv run main.py
```

If you prefer an existing interpreter:

```console
$ uv sync
$ .venv/bin/python main.py
```

### Julia

From this directory, use the repository's project:

```console
$ julia --project=.. main.jl
```

Equivalently, from the repository root:

```console
$ julia --project=. examples/main.jl
```

The first run resolves NumPy for PythonCall through CondaPkg (see the
repository's `CondaPkg.toml`), so no manual Python setup is required.

> Note: the Julia and Python sides may use different Python interpreters — the
> Julia side uses the interpreter managed by PythonCall/CondaPkg, the Python
> side uses the uv virtual environment. The results still match because both
> run the same algorithms with the same seeds.

## The three backends

`example2` and `example3` are both NumPy, but they use **different algorithms**,
so their numbers differ even for the same seed:

| Backend | Python API | Underlying generator |
| --- | --- | --- |
| `PythonRandom` | `random.Random` | CPython Mersenne Twister |
| `NumPyRandom` | `numpy.random.RandomState` | MT19937 (`np.random.seed`) |
| `NumPyRandomDefaultRNG` | `numpy.random.default_rng` | PCG64 (`Generator`) |

`PythonRNGs` mirrors each one draw-for-draw, which is why `main.jl` can line up
with `main.py` regardless of which NumPy API is being used.

## The code

`main.jl`:

```julia
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
```

`main.py`:

```python
import random
import numpy as np

np.set_printoptions(precision=17)

def example1(seed):
	rng = random.Random(seed)
	return [[rng.random() for _ in range(3)] for _ in range(2)]

def example2(seed):
	rng = np.random.RandomState(seed)
	return rng.random((2, 3))

def example3(rng):
	return rng.random((2, 3))

def main():
	print("example1(1234) =")
	print(example1(1234))

	print("example2(999) =")
	print(example2(999))

	rng = np.random.default_rng(42)
	print("example3(rng) =")
	print(example3(rng))

if __name__ == "__main__":
	main()
```
