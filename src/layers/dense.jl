"""
A dense layer with the given input and output size.
- `in` is the input size
- `out` is the output size
- `bias` is a boolean indicating whether to include bias in the layer
- `activation` is the activation function to apply to the output of the layer,
  should be a function that returns `VectorOperator`
- `init` is the function to initialize the weights of the layer
"""
mutable struct Dense{T} <: Layer{T}
    activation::Function
    W::MatrixNode{T}
    b::MatrixNode{T}
    in::Integer
    out::Integer

    Dense{T}(
        (in, out)::Pair{<:Integer, <:Integer};
        bias::Bool=true,
        activation::Function=identity,
        init::Function=zeros
    ) where T = Dense{T}(MatrixVariable(init(T, out, in), name="W"), bias, activation, in, out)

    function Dense{T}(W::MatrixNode{T}, bias::Bool, activation::Function, in::Integer, out::Integer) where T
        b = bias ? MatrixVariable(zeros(T, out, 1), name="b") : MatrixConstant(zeros(T, out, 1), name="b")
        new{T}(activation, W, b, in, out)
    end
end

"""
Creates dense layer computational graph node.
- `x` is the input to the layer
"""
function (layer::Dense)(x::MatrixNode)
    return layer.activation(_dense(x, layer.W, layer.b))
end

_dense(x::MatrixNode{T}, W::MatrixNode{T}, b::MatrixNode{T}) where T = VectorOperator(
    (x::MatrixNode{T}, W::MatrixNode{T}, b::MatrixNode{T}) -> W.value * x.value .+ b.value,
    Function[

        # df/dx
        (x::MatrixNode{T}, W::MatrixNode{T}, b::MatrixNode{T}, g::AbstractVecOrMat{T}) -> W.value' * g,
        
        # df/dW
        (x::MatrixNode{T}, W::MatrixNode{T}, b::MatrixNode{T}, g::AbstractVecOrMat{T}) -> g * x.value',

        # df/db
        (x::MatrixNode{T}, W::MatrixNode{T}, b::MatrixNode{T}, g::AbstractVecOrMat{T}) -> sum(g, dims=2),
    ],
    MatrixNode{T}[x, W, b];
    name="Dense"
)