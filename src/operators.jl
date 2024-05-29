include("graph.jl")

import Base: +, -, *, /, sin, cos, ^

+(x::Node{T}, y::Node{T}) where T = ScalarOperator(
    (x::Node{T}, y::Node{T}) -> x.value + y.value,
    Function[(x::Node{T}, y::Node{T}, g::T) -> 1, (x::Node{T}, y::Node{T}, g::T) -> 1],
    Node{T}[x, y]
)

-(x::Node{T}, y::Node{T}) where T = ScalarOperator(
    (x::Node{T}, y::Node{T}) -> x.value - y.value,
    Function[(x::Node{T}, y::Node{T}, g::T) -> 1, (x::Node{T}, y::Node{T}, g::T) -> -1],
    Node{T}[x, y]
)

*(x::Node{T}, y::Node{T}) where T = ScalarOperator(
    (x::Node{T}, y::Node{T}) -> x.value * y.value,
    Function[(x::Node{T}, y::Node{T}, g::T) -> y.value, (x::Node{T}, y::Node{T}, g::T) -> x.value],
    Node{T}[x, y]
)

/(x::Node{T}, y::Node{T}) where T = ScalarOperator(
    (x::Node{T}, y::Node{T}) -> x.value / y.value,
    Function[(x::Node{T}, y::Node{T}, g::T) -> 1/y.value, (x::Node{T}, y::Node{T}, g::T) -> -x.value/(y.value^2)],
    Node{T}[x, y]
)

sin(x::Node{T}) where T = ScalarOperator(
    (x::Node{T}) -> sin(x.value),
    Function[(x::Node{T}, g::T) -> g * cos(x.value)],
    Node{T}[x]
)

^(x::Node{T}, n::Node{T}) where T = ScalarOperator(
    (x::Node{T}, n::Node{T}) -> x.value ^ n.value,
    Function[(x::Node{T}, n::Node{T}, g::T) -> g * n.value * x.value^(n.value-1), (x::Node{T}, n::Node{T}, g::T) -> g * x.value^n.value * log(abs(x.value))],
    Node{T}[x, n]
)