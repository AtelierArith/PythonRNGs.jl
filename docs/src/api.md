# API

## Backends

```@docs
AbstractPythonRNG
PythonRandom
NumPyRandomDefaultRNG
NumPyRandom
```

## Random interface

```@docs
Random.seed!
Base.copy
```

The backends implement `rand`/`rand!`, `randn`/`randn!`, `randperm`, and
vector `shuffle`/`shuffle!`. Normal and uniform arrays use C-order assignment;
see [Reproducibility](@ref) for backend mappings and type conversions.

## Errors

```@docs
NotSupportedError
```
