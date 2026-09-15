using Aqua
using PythonRNGs

@testset "Aqua quality checks" begin
    Aqua.test_all(PythonRNGs)
end
