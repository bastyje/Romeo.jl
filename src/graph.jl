abstract type Node{T} end
abstract type Operator{T} <: Node{T} end

struct Constant{T} <: Node{T}
    value::T
end

mutable struct Variable{T} <: Node{T}
    value::T
    ∇::Union{Nothing, T}
    Variable(value::T) where T = new{T}(value, nothing)
end

mutable struct ScalarOperator{T} <: Operator{T}
    f::Function
    df::Vector{Function}
    value::Union{Nothing, T}
    inputs::Array{Node{T}, 1}
    ∇::Union{Nothing, T}
    ScalarOperator(f::Function, df::Vector{Function}, inputs::Vector{Node{T}}) where T = new{T}(f, df, nothing, inputs, nothing)
end

function forward!(node::Node{T}) where T
    _forward!(node)
    return node.value
end

_forward!(node::Constant{T}) where T = node.value
_forward!(node::Variable{T}) where T = node.value

function _forward!(node::Operator{T}) where T
    if node.value === nothing
        map(_forward!, node.inputs)
        node.value = node.f(node.inputs...)
    end
    return node
end

update!(node::Constant{T}, ∇) where T = nothing

function update!(node::Node{T}, ∇) where T
    if isnothing(node.∇)
        node.∇ = ∇
    else
        node.∇ += ∇
    end    
end

_backward!(node::Constant{T}) where T = nothing
_backward!(node::Variable{T}) where T = nothing

function _backward!(node::Operator{T}) where T
    for (input, df) in zip(node.inputs, node.df)
        update!(input, df(node.inputs..., node.∇))
        _backward!(input)
    end
end

function backward!(node::Node{T}, seed::T=one(T)) where T
    node.∇ = seed
    _backward!(node)
end