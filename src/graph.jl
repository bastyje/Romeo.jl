"""
Represents node in computational graph.
"""
abstract type Node{T} end

"""
Represents scalar node in computational graph.
"""
abstract type ScalarNode{T} <: Node{T} end

"""
Represents multi-dimensional node in computational graph.
"""
abstract type MatrixNode{T} <: Node{AbstractMatrix{T}} end

"""
Represents constant scalar node in computational graph.
- `value` is the value of the constant.
"""
struct ScalarConstant{T} <: ScalarNode{T}
    value::T
    ScalarConstant(value::T) where T = new{T}(value)
end

"""
Represents constant multi-dimensional node in computational graph.
- `value` is the value of the constant.
"""
struct MatrixConstant{T} <: MatrixNode{T}
    value::AbstractVecOrMat{T}
end

"""
Represents variable scalar node in computational graph.
- `value` is the value of the variable.
- `∇` is the gradient of the variable.
"""
mutable struct ScalarVariable{T} <: ScalarNode{T}
    value::T
    ∇::Union{Nothing, T}
    ScalarVariable(value::T) where T = new{T}(value, nothing)
end

"""
Represents variable multi-dimensional node in computational graph.
- `value` is the value of the variable.
- `∇` is the gradient of the variable.
"""
mutable struct MatrixVariable{T} <: MatrixNode{T}
    value::AbstractVecOrMat{T}
    ∇::Union{Nothing, AbstractVecOrMat{T}}
    MatrixVariable(value::AbstractVecOrMat{T}) where T = new{T}(value, nothing)
end

"""
Represents scalar operator node in computational graph.
- `f` is the function that the operator applies.
- `df` is the list of derivative functions of the operator with respect to each input.
- `value` is calculated value of the operator.
- `inputs` is the list of input nodes.
- `∇` is calculated gradient of the operator.
"""
mutable struct ScalarOperator{T} <: ScalarNode{T}
    f::Function
    df::AbstractVector{Function}
    value::Union{Nothing, T}
    inputs::AbstractVector{ScalarNode{T}}
    ∇::Union{Nothing, T}
    ScalarOperator(f::Function, df::AbstractVector{Function}, inputs::AbstractVector{ScalarNode{T}}) where T = new{T}(f, df, nothing, inputs, nothing)
end

"""
Represents vector operator node in computational graph.
- `f` is the function that the operator applies.
- `df` is the list of derivative functions of the operator with respect to each input.
- `value` is calculated value of the operator.
- `inputs` is the list of input nodes.
- `∇` is calculated gradient of the operator.
"""
mutable struct VectorOperator{T} <: MatrixNode{T}
    f::Function
    df::AbstractVector{Function}
    value::Union{Nothing, AbstractVecOrMat{T}}
    inputs::AbstractVector{MatrixNode{T}}
    ∇::Union{Nothing, AbstractVecOrMat{T}}
    VectorOperator(f::Function, df::AbstractVector{Function}, inputs::AbstractVector{MatrixNode{T}}) where T = new{T}(f, df, nothing, inputs, nothing)
end

"""
Calculates the forward pass of the node
- `node` is the node to calculate the forward pass.
"""
function forward!(node::Node{T}) where T
    _forward!(node)
    return node.value
end

_forward!(node::ScalarConstant{T}) where T = node.value
_forward!(node::MatrixConstant{T}) where T = node.value
_forward!(node::ScalarVariable{T}) where T = node.value
_forward!(node::MatrixVariable{T}) where T = node.value

function _forward!(node::Union{ScalarOperator{T}, VectorOperator{T}}) where T
    if node.value === nothing
        map(_forward!, node.inputs)
        node.value = node.f(node.inputs...)
    end
    return node
end

"""
Updates the gradient of the `ScalarConstant` node
- `node` is the node to update the gradient.
- `∇` is the gradient to update.
"""
update!(::ScalarConstant{T}, ∇) where T = nothing

"""
Updates the gradient of the `MatrixConstant` node
- `node` is the node to update the gradient.
- `∇` is the gradient to update.
"""
update!(::MatrixConstant{T}, ∇) where T = nothing

"""
Updates the gradient of the `Node`
- `node` is the node to update the gradient.
- `∇` is the gradient to update.
"""
function update!(node::Node{T}, ∇) where T
    if isnothing(node.∇)
        node.∇ = ∇
    else
        node.∇ += ∇
    end    
end

_backward!(::ScalarConstant{T}) where T = nothing
_backward!(::MatrixConstant{T}) where T = nothing
_backward!(::ScalarVariable{T}) where T = nothing
_backward!(::MatrixVariable{T}) where T = nothing

function _backward!(node::Union{ScalarOperator{T}, VectorOperator{T}}) where T
    for (input, df) in zip(node.inputs, node.df)
        update!(input, df(node.inputs..., node.∇))
        _backward!(input)
    end
end

"""
Calculates the backward pass of the node
- `node` is the node to calculate the backward pass.
- `seed` is the seed value to start the backward pass.
"""
function backward!(node::ScalarNode{T}, seed::T=one(T)) where T
    node.∇ = seed
    _backward!(node)
end

"""
Calculates the backward pass of the node
- `node` is the node to calculate the backward pass.
- `seed` is the seed value to start the backward pass.
"""
function backward!(node::MatrixNode{T}, seed::AbstractVector{T} = ones(T, size(node.value))) where T
    node.∇ = seed
    _backward!(node)
end