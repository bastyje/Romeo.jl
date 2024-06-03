using Base

"""
Returns node that represents tanh activation function applied to the input node.
- `x` is the input node.
"""
tanh(x::MatrixNode{T}) where T = VectorOperator(
    (x::MatrixNode{T}) -> Base.tanh.(x.value),
    Function[(x::MatrixNode{T}, g::AbstractVecOrMat{T}) -> g .* (1 .- Base.tanh.(x.value).^2)],
    MatrixNode{T}[x];
    name="tanh"
)

"""
Returns node that represents identity activation function applied to the input node.
- `x` is the input node.
"""
identity(x::MatrixNode{T}) where T = VectorOperator(
    (x::MatrixNode{T}) -> x.value,
    Function[(x::MatrixNode{T}, g::AbstractVecOrMat{T}) -> g],
    MatrixNode{T}[x];
    name="identity"
)