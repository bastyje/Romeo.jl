using Test
using autodif: Variable, Constant, forward!, backward!, Dense, Network

@testset "Graph test" begin
    x = Variable(5.0)
    two = Constant(2.0)
    squared = x^two
    sine = sin(squared)
    forward!(sine)
    backward!(sine, 1.0)
    @test sine.value ≈ -0.13235175009777303
    @test x.∇ ≈ 9.912028118634735
end

@testset "Layers tests" begin
    @testset "Dense layer test" begin
        dense = Dense{Float64}(4 => 3)
        @test size(dense.W) == (4, 3)
    end

    @testset "Network tests" begin
        net = Network{Float64}(
            Dense{Float64}(4 => 3),
            Dense{Float64}(3 => 2)
        )
        @test length(net.layers) == 2
        @test net([1.0; 2.0; 3.0]) ≈ [0.0, 0.0]
    end
end