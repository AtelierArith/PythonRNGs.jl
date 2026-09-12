# PythonRNGs examples

The same three examples written twice:

- [`main.py`](main.py) uses Python's `random` and `numpy` directly.
- [`main.jl`](main.jl) uses `PythonRNGs`, and produces **exactly the same
  numbers** by driving the same Python generators.

These examples are not about speed: `PythonRNGs` calls into Python, so it is
slower than a native Julia RNG. They show how to **validate a Julia port of
Python code** by checking that the translated routine matches the original
Python implementation value-for-value.

| Example | Python (`main.py`) | Julia (`main.jl`) | Seed |
| --- | --- | --- | --- |
| `example1` | `random.seed` + `random.random()` | `PythonRandom` | 1234 |
| `example2` | `np.random.seed` + `np.random.random()` (legacy) | `NumPyRandom` | 999 |
| `example3` | `np.random.default_rng` | `NumPyRandomDefaultRNG` | 42 |

## Expected output

Julia:

```console
$ julia --project=.. main.jl
example1(rng) = 2.9329070713842773
example2(rng) = 2.6068560801593756
example3(rng) = 2.5479120971119267
```

Python:

```console
$ uv run main.py
example1()=2.9329070713842773
example2()=2.6068560801593756
example3(rng)=2.5479120971119267
```

The values and the draw order match for every example.

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

example1(seed) = 2rand(PythonRandom(seed)) + 1
example2(seed) = 2rand(NumPyRandom(seed)) + 1
example3(rng::Random.AbstractRNG) = 2rand(rng) + 1

@show example1(1234)
@show example2(999)

rng = NumPyRandomDefaultRNG(42)
@show example3(rng)
```

`main.py`:

```python
import random
import numpy as np

def example1(seed):
    random.seed(seed)
    return 2 * random.random() + 1

def example2(seed):
    np.random.seed(seed)
    return 2 * np.random.random() + 1

def example3(rng):
    return 2 * rng.random() + 1

def main():
    print(f"{example1(1234)=}")
    print(f"{example2(999)=}")
    rng = np.random.default_rng(42)
    print(f"{example3(rng)=}")

if __name__ == "__main__":
    main()
```
