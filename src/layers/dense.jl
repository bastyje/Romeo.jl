include("network.jl")

"""
A dense layer with the given input and output size.
- `in` is the input size
- `out` is the output size
- `bias` is a boolean indicating whether to include bias in the layer
- `activation` is the activation function to apply to the output of the layer
- `init` is the function to initialize the weights of the layer
"""
mutable struct Dense{T} <: Layer{T}
    activation::Function
    W::AbstractMatrix{T}
    b::AbstractVector{T}
    in::Integer
    out::Integer

    Dense{T}(
        (in, out)::Pair{<:Integer, <:Integer};
        bias::Bool=true,
        activation::Function=identity,
        init::Function=zeros
    ) where T = Dense{T}(init(out, in), bias, activation, in, out)

    function Dense{T}(W::AbstractMatrix, bias::Bool, activation::Function, in::Integer, out::Integer) where T
        b = bias ? zeros(T, size(W, 1)) : nothing # todo
        new{T}(activation, W, b, in, out)
    end
end

"""
Applies the dense layer to the input `x`.
- `x` is the input to the dense layer
"""
function (layer::Dense)(x::AbstractArray)
    return layer.activation.(layer.W * x .+ layer.b)
end