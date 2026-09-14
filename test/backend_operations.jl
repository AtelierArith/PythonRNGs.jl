normal_reference(ref, R) = R === PythonRandom ? ref.gauss(0, 1) : ref.standard_normal()

@testset "Backend normal draws" for R in (PythonRandom, NumPyRandomDefaultRNG, NumPyRandom), seed in (0, 42)
    rng = R(seed)
    ref = rng.factory(seed)
    # Interleave scalars, odd/even arrays, empty arrays, and uniform draws so
    # cached normals and backend state must survive all entry points.
    for dims in ((), (1,), (5,), (2, 3), (2, 3, 4), (0,), (2, 0, 3))
        @test randn(rng) == pyconvert(Float64, normal_reference(ref, R))
        actual = randn(rng, Float64, dims)
        expected = if R === PythonRandom
            flat = [pyconvert(Float64, ref.gauss(0, 1)) for _ in 1:prod(dims)]
            pyimport("numpy").array(flat).reshape(dims)
        else
            ref.standard_normal(size = dims)
        end
        @test actual isa Array{Float64,length(dims)}
        @test actual == pyconvert(Array, expected)
        for I in CartesianIndices(actual)
            @test actual[I] == pyconvert(Float64, expected[(Tuple(I) .- 1)...])
        end
        @test rand(rng) == pyconvert(Float64, ref.random())
        @test randn(rng) == pyconvert(Float64, normal_reference(ref, R))
        @test randn(R(seed), dims) == randn(R(seed), Float64, dims)
        if !isempty(dims)
            @test randn(R(seed), dims...) == randn(R(seed), Float64, dims...)
        end
    end
    for T in (Float16, Float32, Float64)
        rng, ref = R(seed), R(seed).factory(seed)
        for _ in 1:5
            expected = T === Float32 && R === NumPyRandomDefaultRNG ?
                ref.standard_normal(dtype = "float32") : normal_reference(ref, R)
            @test randn(rng, T) === T(pyconvert(Float64, expected))
        end
        A = zeros(T, 3, 5)
        @test randn!(rng, A) === A
        if T === Float32 && R === NumPyRandomDefaultRNG
            @test A == pyconvert(Array, ref.standard_normal(size = (3, 5), dtype = "float32"))
        else
            expected = [T(pyconvert(Float64, normal_reference(ref, R))) for _ in 1:15]
            @test [A[i,j] for i in 1:3 for j in 1:5] == expected
        end
        @test rand(rng) == pyconvert(Float64, ref.random())
    end
    rng = R(seed)
    randn(rng) # Leave any cached second normal pending.
    c, d = copy(rng), deepcopy(rng)
    @test randn(rng, 7) == randn(c, 7) == randn(d, 7)
    Random.seed!(rng, seed)
    @test randn(rng, 7) == randn(R(seed), 7)
    parent = fill(-99.0, 4, 6)
    dest = @view parent[1:2:4, 2:2:6]
    @test randn!(R(seed), dest) === dest
    @test dest == randn(R(seed), 2, 3)
    @test all(==(-99), parent[2:2:4, :])
    @test all(==(-99), parent[:, 1:2:5])
    rng = R(seed)
    @test_throws ArgumentError randn(rng, 2, -1)
    @test randn(rng) == randn(R(seed))
end

@testset "Backend permutations and shuffles" for R in (PythonRandom, NumPyRandomDefaultRNG, NumPyRandom), seed in (0, 42)
    for n in (0, 1, 2, 5, 20, 100)
        rng = R(seed)
        ref = rng.factory(seed)
        expected = R === PythonRandom ? ref.sample(pybuiltins.range(n), n) : ref.permutation(n)
        actual = randperm(rng, n)
        @test actual == pyconvert(Vector{Int}, expected) .+ 1
        @test sort(actual) == collect(1:n)
        @test rand(rng) == pyconvert(Float64, ref.random())
        @test randn(rng) == pyconvert(Float64, normal_reference(ref, R))

        for values in (collect(1:n), [isodd(i) for i in 1:n], ["v$i" for i in 1:n])
            rng = R(seed)
            ref = rng.factory(seed)
            pyvalues = R === PythonRandom ? pybuiltins.list(values) : pyimport("numpy").array(values)
            ref.shuffle(pyvalues)
            original = copy(values)
            result = shuffle(rng, values)
            @test values == original
            @test result !== values
            @test result == pyconvert(Vector{eltype(values)}, pyvalues)
            @test rand(rng) == pyconvert(Float64, ref.random())
            @test shuffle!(R(seed), values) === values
            @test values == result
        end
    end
    @test eltype(randperm(R(seed), Int32(5))) === Int32
    rng = R(seed)
    @test_throws ArgumentError randperm(rng, -1)
    @test rand(rng) == rand(R(seed))
    @test shuffle(R(seed), Base.OneTo(20)) == shuffle(R(seed), collect(1:20))
    parent = collect(1:12)
    dest = @view parent[1:2:11]
    @test shuffle!(R(seed), dest) === dest
    @test dest == shuffle(R(seed), collect(1:2:11))
    @test parent[2:2:12] == collect(2:2:12)
    objects = [Ref(i) for i in 1:8]
    order = shuffle(R(seed), collect(1:8))
    result = shuffle(R(seed), objects)
    @test all(result[i] === objects[order[i]] for i in 1:8)
end

@testset "Legacy sampling without replacement" begin
    rng = NumPyRandom(42)
    ref = rng.factory(42)
    @test randperm(rng, 20)[1:7] == pyconvert(Vector{Int}, ref.choice(20, 7, replace = false)) .+ 1
    @test rand(rng) == pyconvert(Float64, ref.random())
end

@testset "Random method compatibility" begin
    @test isempty(Test.detect_ambiguities(PythonRNGs, Random))
end

@testset "Complex array dispatch and C order" for R in (PythonRandom, NumPyRandomDefaultRNG, NumPyRandom), dims in ((), (2, 3), (2, 3, 4))
    rng, ref = R(42), R(42)
    A = Array{ComplexF64}(undef, dims)
    @test rand!(rng, A) === A
    expected = [rand(ref, ComplexF64) for _ in 1:prod(dims)]
    @test vec(permutedims(A, reverse(ntuple(identity, length(dims))))) == expected
    @test rand(rng) == rand(ref)
    @test A == rand(R(42), ComplexF64, dims)
end
