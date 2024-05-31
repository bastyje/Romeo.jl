include("graph.jl")

import Base: +, -, *, /, sin, cos, ^

+(x::ScalarNode{T}, y::ScalarNode{T}) where T = ScalarOperator(
    (x::ScalarNode{T}, y::ScalarNode{T}) -> x.value + y.value,
    Function[(x::ScalarNode{T}, y::ScalarNode{T}, g::T) -> 1, (x::ScalarNode{T}, y::ScalarNode{T}, g::T) -> 1],
    ScalarNode{T}[x, y]
)

-(x::ScalarNode{T}, y::ScalarNode{T}) where T = ScalarOperator(
    (x::ScalarNode{T}, y::ScalarNode{T}) -> x.value - y.value,
    Function[(x::ScalarNode{T}, y::ScalarNode{T}, g::T) -> 1, (x::ScalarNode{T}, y::ScalarNode{T}, g::T) -> -1],
    ScalarNode{T}[x, y]
)

*(x::ScalarNode{T}, y::ScalarNode{T}) where T = ScalarOperator(
    (x::ScalarNode{T}, y::ScalarNode{T}) -> x.value * y.value,
    Function[(x::ScalarNode{T}, y::ScalarNode{T}, g::T) -> y.value, (x::ScalarNode{T}, y::ScalarNode{T}, g::T) -> x.value],
    ScalarNode{T}[x, y]
)

/(x::ScalarNode{T}, y::ScalarNode{T}) where T = ScalarOperator(
    (x::ScalarNode{T}, y::ScalarNode{T}) -> x.value / y.value,
    Function[(x::ScalarNode{T}, y::ScalarNode{T}, g::T) -> 1/y.value, (x::ScalarNode{T}, y::ScalarNode{T}, g::T) -> -x.value/(y.value^2)],
    ScalarNode{T}[x, y]
)

sin(x::ScalarNode{T}) where T = ScalarOperator(
    (x::ScalarNode{T}) -> sin(x.value),
    Function[(x::ScalarNode{T}, g::T) -> g * cos(x.value)],
    ScalarNode{T}[x]
)

^(x::ScalarNode{T}, n::ScalarNode{T}) where T = ScalarOperator(
    (x::ScalarNode{T}, n::ScalarNode{T}) -> x.value ^ n.value,
    Function[(x::ScalarNode{T}, n::ScalarNode{T}, g::T) -> g * n.value * x.value^(n.value-1), (x::ScalarNode{T}, n::ScalarNode{T}, g::T) -> g * x.value^n.value * log(abs(x.value))],
    ScalarNode{T}[x, n]
)

+(x::MatrixNode{T}, y::MatrixNode{T}) where T = VectorOperator(
    (x::MatrixNode{T}, y::MatrixNode{T}) -> x.value + y.value,
    Function[(x::MatrixNode{T}, y::MatrixNode{T}, g::AbstractVecOrMat{T}) -> g, (x::MatrixNode{T}, y::MatrixNode{T}, g::AbstractVecOrMat{T}) -> g],
    MatrixNode{T}[x, y]
)

-(x::MatrixNode{T}, y::MatrixNode{T}) where T = VectorOperator(
    (x::MatrixNode{T}, y::MatrixNode{T}) -> x.value - y.value,
    Function[(x::MatrixNode{T}, y::MatrixNode{T}, g::AbstractVecOrMat{T}) -> g, (x::MatrixNode{T}, y::MatrixNode{T}, g::AbstractVecOrMat{T}) -> -g],
    MatrixNode{T}[x, y]
)

*(x::MatrixNode{T}, y::MatrixNode{T}) where T = VectorOperator(
    (x::MatrixNode{T}, y::MatrixNode{T}) -> x.value * y.value,
    Function[(x::MatrixNode{T}, y::MatrixNode{T}, g::AbstractVecOrMat{T}) -> g * y.value', (x::MatrixNode{T}, y::MatrixNode{T}, g::AbstractVecOrMat{T}) -> x.value' * g],
    MatrixNode{T}[x, y]
)

/(x::MatrixNode{T}, y::MatrixNode{T}) where T = VectorOperator(
    (x::MatrixNode{T}, y::MatrixNode{T}) -> x.value / y.value,
    Function[(x::MatrixNode{T}, y::MatrixNode{T}, g::AbstractVecOrMat{T}) -> g / y.value, (x::MatrixNode{T}, y::MatrixNode{T}, g::AbstractVecOrMat{T}) -> -g * x.value / y.value^2],
    MatrixNode{T}[x, y]
)

