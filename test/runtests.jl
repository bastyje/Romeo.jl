using Test
using autodif
using Test
using Random
using Combinatorics

Random.seed!(1234)

@testset "Graph test" begin
    @testset "Scalars" begin
        x = autodif.ScalarVariable(5.0)
        two = autodif.ScalarConstant(2.0)
        squared = x^two
        sine = sin(squared)
        autodif.forward!(sine)
        autodif.backward!(sine, 1.0)
        @test sine.value ≈ -0.13235175009777303
        @test x.∇ ≈ 9.912028118634735
    end

    @testset "Matrices" begin
        A = autodif.MatrixVariable([1.0 2.0; 3.0 4.0])
        B = autodif.MatrixConstant([5.0; 6.0])
        C = A * B
        autodif.forward!(C)
        autodif.backward!(C)
        @test C.value ≈ [17.0; 39.0]
        @test A.∇ ≈ [5.0 6.0; 5.0 6.0]
    end

    @testset "ScalarConstant" begin
        sc = autodif.ScalarConstant(5)
        @test autodif.forward!(sc) == 5
        @test sc.value == 5
    end

    @testset "MatrixConstant" begin
        mc = autodif.MatrixConstant([1 2; 3 4])
        @test autodif.forward!(mc) == [1 2; 3 4]
        @test mc.value == [1 2; 3 4]
    end

    @testset "ScalarVariable" begin
        sv = autodif.ScalarVariable(10)
        @test autodif.forward!(sv) == 10
        @test sv.value == 10
        sv.value = 20
        @test autodif.forward!(sv) == 20
        @test sv.value == 20
    end

    @testset "MatrixVariable" begin
        mv = autodif.MatrixVariable([5 6; 7 8])
        @test autodif.forward!(mv) == [5 6; 7 8]
        @test mv.value == [5 6; 7 8]
        mv.value = [9 10; 11 12]
        @test autodif.forward!(mv) == [9 10; 11 12]
        @test mv.value == [9 10; 11 12]
    end

    @testset "ScalarOperator" begin
        so = autodif.ScalarOperator(
            (x::autodif.ScalarNode, y::autodif.ScalarNode) -> x.value + y.value,
            Function[(x::autodif.ScalarNode, y::autodif.ScalarNode, g::Int64) -> 1, (x::autodif.ScalarNode, y::autodif.ScalarNode, g::Int64) -> 1],
            autodif.ScalarNode{Int64}[autodif.ScalarConstant(1), autodif.ScalarConstant(2)]
        )
        @test autodif.forward!(so) == 3
        @test so.value == 3
        
        so = autodif.ScalarOperator(
            (x::autodif.ScalarNode, y::autodif.ScalarNode) -> x.value * y.value,
            Function[(x::autodif.ScalarNode, y::autodif.ScalarNode, g::Int64) -> y.value, (x::autodif.ScalarNode, y::autodif.ScalarNode, g::Int64) -> x.value],
            autodif.ScalarNode{Int64}[autodif.ScalarConstant(2), autodif.ScalarConstant(3)]
        )
        @test autodif.forward!(so) == 6
        @test so.value == 6
    end

    @testset "VectorOperator" begin
        vo = autodif.VectorOperator(
            (x::autodif.MatrixNode, y::autodif.MatrixNode) -> x.value + y.value,
            Function[(x::autodif.MatrixNode, y::autodif.MatrixNode, g::AbstractVecOrMat{Int64}) -> g, (x::autodif.MatrixNode, y::autodif.MatrixNode, g::AbstractVecOrMat{Int64}) -> g],
            autodif.MatrixNode{Int64}[autodif.MatrixConstant([1 2; 3 4]), autodif.MatrixConstant([5 6; 7 8])]
        )
        @test autodif.forward!(vo) == [6 8; 10 12]
        @test vo.value == [6 8; 10 12]
        
        vo = autodif.VectorOperator(
            (x::autodif.MatrixNode, y::autodif.MatrixNode) -> x.value * y.value,
            Function[(x::autodif.MatrixNode, y::autodif.MatrixNode, g::AbstractVecOrMat{Int64}) -> g * y.value', (x::autodif.MatrixNode, y::autodif.MatrixNode, g::AbstractVecOrMat{Int64}) -> x.value' * g],
            autodif.MatrixNode{Int64}[autodif.MatrixConstant([1 2; 3 4]), autodif.MatrixConstant([5 6; 7 8])]
        )
        @test autodif.forward!(vo) == [19 22; 43 50]
        @test vo.value == [19 22; 43 50]
    end

    @testset "ScalarOperator backward pass" begin
        so = autodif.ScalarOperator(
            (x::autodif.ScalarNode, y::autodif.ScalarNode) -> x.value * y.value,
            Function[(x::autodif.ScalarNode, y::autodif.ScalarNode, g::Int64) -> y.value, (x::autodif.ScalarNode, y::autodif.ScalarNode, g::Int64) -> x.value],
            autodif.ScalarNode{Int64}[autodif.ScalarVariable(1), autodif.ScalarConstant(2)]
        )
        autodif.forward!(so)
        autodif.backward!(so)
        @test so.∇ == 1
        @test so.inputs[1].∇ == 2
    end

    @testset "VectorOperator backward pass" begin
        vo = autodif.VectorOperator(
            (x::autodif.MatrixNode, y::autodif.MatrixNode) -> x.value * y.value,
            Function[(x::autodif.MatrixNode, y::autodif.MatrixNode, g::AbstractVecOrMat{Int64}) -> g * y.value', (x::autodif.MatrixNode, y::autodif.MatrixNode, g::AbstractVecOrMat{Int64}) -> x.value' * g],
            autodif.MatrixNode{Int64}[autodif.MatrixVariable([1 2; 3 4]), autodif.MatrixConstant([5; 6])]
        )
        autodif.forward!(vo)
        autodif.backward!(vo)
        @test vo.∇ == ones(size(vo.value))
        @test vo.inputs[1].∇ == [5 6; 5 6]
    end
end

@testset "Layers tests" begin
    @testset "Dense Layer" begin
        Random.seed!(123)
        dense = autodif.Dense{Float64}(3 => 2; activation = identity, init = rand)
        x = rand(3)
        @test size(dense(x)) == (2,)
        @test dense.in == 3
        @test dense.out == 2
    end

    @testset "RNNCell" begin
        Random.seed!(123)
        rnn_cell = autodif.RNNCell{Float64}(3 => 2, activation = identity, init = rand)
        x = rand(3)
        state = rand(2)
        @test size(rnn_cell(x, state)) == (2,)
        @test rnn_cell.in == 3
        @test rnn_cell.out == 2
    end

    @testset "RNN Layer" begin
        Random.seed!(123)
        rnn_cell = autodif.RNNCell{Float64}(3 => 2, activation = identity, init = rand)
        rnn = autodif.RNN{Float64}(rnn_cell)
        x = rand(3)
        @test size(rnn(x)) == (2,)
    end

    @testset "Network with Dense Layer" begin
        Random.seed!(123)
        dense1 = autodif.Dense{Float64}(3 => 2, activation = identity, init = rand)
        dense2 = autodif.Dense{Float64}(2 => 1, activation = identity, init = rand)
        network = autodif.Network(dense1, dense2)
        x = rand(3)
        @test size(network(x)) == (1,)
    end

    @testset "Network with RNN Layer" begin
        Random.seed!(123)
        rnn_cell = autodif.RNNCell{Float64}(3 => 2, activation = identity, init = rand)
        rnn = autodif.RNN{Float64}(rnn_cell)
        dense = autodif.Dense{Float64}(2 => 1, activation = identity, init = rand)
        network = autodif.Network(rnn, dense)
        x = rand(3)
        @test size(network(x)) == (1,)
    end

    @testset "Network with Dense and RNN Layers" begin
        Random.seed!(123)
        dense1 = autodif.Dense{Float64}(3 => 2, activation = identity, init = rand)
        rnn_cell = autodif.RNNCell{Float64}(2 => 2, activation = identity, init = rand)
        rnn = autodif.RNN{Float64}(rnn_cell)
        dense2 = autodif.Dense{Float64}(2 => 1, activation = identity, init = rand)
        network = autodif.Network(dense1, rnn, dense2)
        x = rand(3)
        @test size(network(x)) == (1,)
    end
end

