using Test
using CondaPkg
using PythonRNGs
using Random
using PythonCall

@testset "PythonRNGs" begin
    @testset "AbstractRNG subtype" begin
        @test PythonRandom <: AbstractPythonRNG
        @test NumPyRandomDefaultRNG <: AbstractPythonRNG
        @test NumPyRandom <: AbstractPythonRNG
        @test AbstractPythonRNG <: AbstractRNG
    end

    @testset "PythonRandom matches random.Random($seed)" for seed in (0, 1234)
        rng = PythonRandom(seed)
        ref = pyimport("random").Random(seed)
        for _ = 1:20
            @test rand(rng) == pyconvert(Float64, ref.random())
        end
    end

    @testset "NumPyRandomDefaultRNG matches numpy.random.default_rng($seed)" for seed in
                                                                                 (0, 1234)
        rng = NumPyRandomDefaultRNG(seed)
        ref = pyimport("numpy").random.default_rng(seed)
        for _ = 1:20
            @test rand(rng) == pyconvert(Float64, ref.random())
        end
    end

    @testset "NumPyRandom matches numpy.random.RandomState($seed)" for seed in (0, 999)
        rng = NumPyRandom(seed)
        ref = pyimport("numpy").random.RandomState(seed)
        for _ = 1:20
            @test rand(rng) == pyconvert(Float64, ref.random_sample())
        end

        # Same as `np.random.seed(seed); np.random.random()`.
        rng = NumPyRandom(seed)
        ref = pyimport("numpy").random.RandomState(seed)
        @test [rand(rng) for _ = 1:5] == [pyconvert(Float64, ref.random_sample()) for _ = 1:5]

        # Integers match the legacy `RandomState.randint`.
        for (T, k) in ((UInt8, 8), (UInt16, 16), (UInt32, 32))
            r = NumPyRandom(seed)
            rf = pyimport("numpy").random.RandomState(seed)
            @test [rand(r, T) for _ = 1:5] == [pyconvert(T, rf.randint(0, 2^k, dtype = "uint" * string(k))) for _ = 1:5]
        end

        # Ranges match `RandomState.randint`.
        r = NumPyRandom(seed)
        rf = pyimport("numpy").random.RandomState(seed)
        @test [rand(r, 1:6) for _ = 1:5] == [pyconvert(Int, rf.randint(1, 7)) for _ = 1:5]

        # Full `UInt64` uses a Python big-int bound.
        r = NumPyRandom(seed)
        rf = pyimport("numpy").random.RandomState(seed)
        @test [rand(r, UInt64) for _ = 1:5] == [pyconvert(UInt64, rf.randint(0, big(2)^64, dtype = "uint64")) for _ = 1:5]

        # Step ranges via `choice`.
        r = NumPyRandom(seed)
        rf = pyimport("numpy").random.RandomState(seed)
        @test [rand(r, 1:2:9) for _ = 1:5] == [pyconvert(Int, rf.choice(collect(1:2:9))) for _ = 1:5]

        @test_throws NotSupportedError rand(NumPyRandom(1), Int128(1):Int128(6))
        @test_throws NotSupportedError rand(NumPyRandom(1), big(1):big(6))
    end

    @testset "seed! reproduces the stream" for rng in (
        PythonRandom(42),
        NumPyRandomDefaultRNG(42),
        NumPyRandom(42),
    )
        a = rand(rng, 10)
        Random.seed!(rng, 42)
        @test rand(rng, 10) == a
    end

    @testset "seed! without an argument reseeds" begin
        rng = PythonRandom(1)
        Random.seed!(rng)
        @test rand(rng) isa Float64
        Random.seed!(rng, nothing)
        @test rand(rng) isa Float64
    end

    @testset "uniform scalar types" for rng in (
        PythonRandom(7),
        NumPyRandomDefaultRNG(7),
        NumPyRandom(7),
    )
        for T in (Float64, Float32, Float16)
            x = rand(rng, T)
            @test x isa T
            @test zero(T) <= x < one(T)
        end
        @test rand(rng, Bool) isa Bool
        for T in (UInt8, UInt16, UInt32, UInt64, UInt128, Int8, Int16, Int32, Int64, Int128)
            @test rand(rng, T) isa T
        end
    end

    @testset "ranges and arrays" for rng in (
        PythonRandom(99),
        NumPyRandomDefaultRNG(99),
        NumPyRandom(99),
    )
        @test rand(rng, 1:6) in 1:6
        v = rand(rng, 1:6, 100)
        @test length(v) == 100
        @test all(x -> x in 1:6, v)
        f = rand(rng, 100)
        @test length(f) == 100
        @test all(x -> 0 <= x < 1, f)
    end

    @testset "instances are independent" begin
        @test rand(PythonRandom(5), 5) == rand(PythonRandom(5), 5)
        @test rand(PythonRandom(1), 5) != rand(PythonRandom(2), 5)
        @test rand(NumPyRandomDefaultRNG(5), 5) == rand(NumPyRandomDefaultRNG(5), 5)
        @test rand(NumPyRandomDefaultRNG(1), 5) != rand(NumPyRandomDefaultRNG(2), 5)
        @test rand(NumPyRandom(5), 5) == rand(NumPyRandom(5), 5)
        @test rand(NumPyRandom(1), 5) != rand(NumPyRandom(2), 5)
    end
    @testset "integer draws are reproducible" for rng in (
        PythonRandom(3),
        NumPyRandomDefaultRNG(3),
        NumPyRandom(3),
    )
        a = [rand(rng, UInt64) for _ = 1:10]
        Random.seed!(rng, 3)
        @test [rand(rng, UInt64) for _ = 1:10] == a
    end

    @testset "integer draws match Python ($seed)" for seed in (0, 2024)
        for (T, k) in ((UInt8, 8), (UInt16, 16), (UInt32, 32), (UInt64, 64), (UInt128, 128))
            rng = PythonRandom(seed)
            ref = pyimport("random").Random(seed)
            @test [rand(rng, T) for _ = 1:5] == [pyconvert(T, ref.getrandbits(k)) for _ = 1:5]
        end
        rng = PythonRandom(seed)
        ref = pyimport("random").Random(seed)
        @test [rand(rng, Bool) for _ = 1:5] == [pyconvert(Bool, ref.getrandbits(1)) for _ = 1:5]

        rng = NumPyRandomDefaultRNG(seed)
        ref = pyimport("numpy").random.default_rng(seed)
        @test [rand(rng, UInt64) for _ = 1:5] == [pyconvert(UInt64, ref.bit_generator.random_raw()) for _ = 1:5]
        for (T, k, dt) in
            ((UInt8, 8, "uint8"), (UInt16, 16, "uint16"), (UInt32, 32, "uint32"))
            rng = NumPyRandomDefaultRNG(seed)
            ref = pyimport("numpy").random.default_rng(seed)
            @test [rand(rng, T) for _ = 1:5] == [pyconvert(T, ref.integers(0, 2^k, dtype = dt)) for _ = 1:5]
        end

        # signed draws reinterpret the native unsigned draw
        for T in (Int8, Int16, Int32, Int64, Int128)
            a = PythonRandom(seed)
            b = PythonRandom(seed)
            U = unsigned(T)
            @test rand(a, T) == rand(b, U) % T
        end
    end

    @testset "array draws match Python ($seed)" for seed in (0, 2024)
        rng = PythonRandom(seed)
        ref = pyimport("random").Random(seed)
        @test rand(rng, 6) == [pyconvert(Float64, ref.random()) for _ = 1:6]

        rng = NumPyRandomDefaultRNG(seed)
        ref = pyimport("numpy").random.default_rng(seed)
        @test rand(rng, 6) == [pyconvert(Float64, x) for x in ref.random(6).tolist()]

        # NumPy has a native Float32
        rng = NumPyRandomDefaultRNG(seed)
        ref = pyimport("numpy").random.default_rng(seed)
        @test [rand(rng, Float32) for _ = 1:5] == [pyconvert(Float32, ref.random(dtype = "float32")) for _ = 1:5]

        rng = NumPyRandom(seed)
        ref = pyimport("numpy").random.RandomState(seed)
        @test rand(rng, 6) == [pyconvert(Float64, x) for x in ref.random_sample(6).tolist()]
    end

    @testset "integer ranges match Python ($seed)" for seed in (0, 2024)
        rng = PythonRandom(seed)
        ref = pyimport("random").Random(seed)
        @test [rand(rng, 1:6) for _ = 1:6] == [pyconvert(Int, ref.randint(1, 6)) for _ = 1:6]

        rng = PythonRandom(seed)
        ref = pyimport("random").Random(seed)
        @test [rand(rng, 1:2:9) for _ = 1:6] == [pyconvert(Int, ref.randrange(1, 10, 2)) for _ = 1:6]

        rng = NumPyRandomDefaultRNG(seed)
        ref = pyimport("numpy").random.default_rng(seed)
        @test [rand(rng, 1:6) for _ = 1:6] == [pyconvert(Int, ref.integers(1, 7)) for _ = 1:6]

        rng = NumPyRandomDefaultRNG(seed)
        ref = pyimport("numpy").random.default_rng(seed)
        @test [rand(rng, UInt8(1):UInt8(6)) for _ = 1:6] == [pyconvert(UInt8, ref.integers(1, 7, dtype = "uint8")) for _ = 1:6]

        rng = NumPyRandomDefaultRNG(seed)
        ref = pyimport("numpy").random.default_rng(seed)
        @test rand(rng, 1:6, 4) ==
              [pyconvert(Int, x) for x in ref.integers(1, 7, size = 4).tolist()]
    end

    @testset "non-integer ranges match choice ($seed)" for seed in (0, 2024)
        rng = PythonRandom(seed)
        ref = pyimport("random").Random(seed)
        @test [rand(rng, 'a':'z') for _ = 1:6] == [pyconvert(String, ref.choice("abcdefghijklmnopqrstuvwxyz"))[1] for _ = 1:6]

        rng = NumPyRandomDefaultRNG(seed)
        ref = pyimport("numpy").random.default_rng(seed)
        @test [rand(rng, 'a':'z') for _ = 1:6] == [
            pyconvert(String, ref.choice(collect("abcdefghijklmnopqrstuvwxyz")))[1] for
            _ = 1:6
        ]

        rng = NumPyRandom(seed)
        ref = pyimport("numpy").random.RandomState(seed)
        @test [rand(rng, 'a':'z') for _ = 1:6] == [
            pyconvert(String, ref.choice(collect("abcdefghijklmnopqrstuvwxyz")))[1] for
            _ = 1:6
        ]

        rng = PythonRandom(seed)
        ref = pyimport("random").Random(seed)
        r = 1.0:0.5:2.0
        @test [rand(rng, r) for _ = 1:6] == [pyconvert(Float64, ref.choice(collect(r))) for _ = 1:6]

        rng = NumPyRandom(seed)
        ref = pyimport("numpy").random.RandomState(seed)
        r = 1.0:0.5:2.0
        @test [rand(rng, r) for _ = 1:6] == [pyconvert(Float64, ref.choice(collect(r))) for _ = 1:6]
    end

    @testset "unsupported ranges throw NotSupportedError" begin
        @test rand(PythonRandom(1), Int128(1):Int128(6)) isa Int128
        @test rand(PythonRandom(1), big(1):big(6)) isa BigInt
        @test_throws NotSupportedError rand(NumPyRandomDefaultRNG(1), Int128(1):Int128(6))
        @test_throws NotSupportedError rand(NumPyRandomDefaultRNG(1), big(1):big(6))
        @test_throws NotSupportedError rand(NumPyRandomDefaultRNG(1), UInt128(1):UInt128(6))
    end

    @testset "multidimensional arrays ($seed)" for seed in (0, 2024)
        # PythonRandom: wrap the flat list in a NumPy array and reshape it in
        # Fortran order (NumPy has no generator for the stdlib `random`).
        rng = PythonRandom(seed)
        x = rand(rng, 2, 3)
        ref = pyimport("random").Random(seed)
        flat = [pyconvert(Float64, ref.random()) for _ = 1:6]
        @test vec(x) == flat
        mF = pyimport("numpy").reshape(pyimport("numpy").array(flat), (2, 3), order = "F")
        @test x == [pyconvert(Float64, mF[i, k]) for i = 0:1, k = 0:2]

        # NumPyRandomDefaultRNG: Julia fills column-major, i.e. NumPy's Fortran order.
        rng = NumPyRandomDefaultRNG(seed)
        y = rand(rng, 2, 3)
        ref = pyimport("numpy").random.default_rng(seed)
        mF = pyimport("numpy").reshape(ref.random(6), (2, 3), order = "F")
        @test y == [pyconvert(Float64, mF[i, k]) for i = 0:1, k = 0:2]

        # NumPyRandom: same Fortran-order layout.
        rng = NumPyRandom(seed)
        z = rand(rng, 2, 3)
        ref = pyimport("numpy").random.RandomState(seed)
        mF = pyimport("numpy").reshape(ref.random_sample(6), (2, 3), order = "F")
        @test z == [pyconvert(Float64, mF[i, k]) for i = 0:1, k = 0:2]

        for r in (PythonRandom(seed), NumPyRandomDefaultRNG(seed), NumPyRandom(seed))
            x = rand(r, 2, 3)
            @test x isa Matrix{Float64}
            @test size(x) == (2, 3)
            @test size(rand(r, Float32, 2, 3)) == (2, 3)
            @test size(rand(r, 1:6, 2, 3)) == (2, 3)
            @test size(rand(r, 'a':'z', 2, 3)) == (2, 3)
        end
    end

    @testset "copy / deepcopy are independent" for rng in (
        PythonRandom(11),
        NumPyRandomDefaultRNG(11),
        NumPyRandom(11),
    )
        c = copy(rng)
        @test rand(rng, 5) == rand(c, 5)
        d = deepcopy(rng)
        @test rand(rng, 5) == rand(d, 5)
    end
end
