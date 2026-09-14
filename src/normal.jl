# Keep the backend's normal cache and stream consumption on the Python side.
Random.randn(rng::AbstractPythonRNG) = randn(rng, Float64)
Random.randn(rng::PythonRandom, ::Type{Float64}) =
    pyconvert(Float64, rng.pyobj.gauss(0, 1))
Random.randn(rng::Union{NumPyRandomDefaultRNG,NumPyRandom}, ::Type{Float64}) =
    pyconvert(Float64, rng.pyobj.standard_normal())
Random.randn(rng::NumPyRandomDefaultRNG, ::Type{Float32}) =
    pyconvert(Float32, rng.pyobj.standard_normal(dtype = "float32"))
Random.randn(rng::AbstractPythonRNG, ::Type{Float32}) = Float32(randn(rng, Float64))
Random.randn(rng::AbstractPythonRNG, ::Type{Float16}) = Float16(randn(rng, Float64))

function Random.randn(rng::AbstractPythonRNG, ::Type{T}, dims::Dims) where {T}
    return randn!(rng, Array{T}(undef, dims))
end

function Random.randn!(rng::AbstractPythonRNG, A::AbstractArray{T}) where {T}
    for I in CartesianIndices(reverse(axes(A)))
        @inbounds A[reverse(Tuple(I))...] = randn(rng, T)
    end
    return A
end

function Random.randn!(
    rng::Union{NumPyRandomDefaultRNG,NumPyRandom}, A::AbstractArray{Float64},
)
    return copyto!(A, pyconvert(Array, rng.pyobj.standard_normal(size = size(A))))
end

function Random.randn!(rng::NumPyRandomDefaultRNG, A::AbstractArray{Float32})
    return copyto!(A, pyconvert(
        Array, rng.pyobj.standard_normal(size = size(A), dtype = "float32"),
    ))
end
