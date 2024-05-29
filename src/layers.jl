abstract type Layer{T} end

struct Dense{T} <: Layer{T}
    activation::Function
    W::AbstractMatrix{T}
    b::AbstractVector{T}

    Dense{T}(
        (in, out)::Pair{<:Integer, <:Integer};
        bias::Bool=true,
        activation::Function=identity,
        init::Function=zeros
    ) where T = Dense{T}(init(in, out), bias, activation)

    function Dense{T}(W::AbstractMatrix, bias::Bool, activation::Function,) where T
        b = bias ? zeros(T, size(W, 1)) : nothing # todo
        new{T}(activation, W, b)
    end
end

function (layer::Dense)(x::AbstractArray)
    return layer.activation.(layer.W * x .+ layer.b)
end

struct Network{T}
    layers::Vector{Layer{T}}

    Network{T}(layers::Layer{T}...) where T = new{T}(collect(layers))
end

function (net::Network)(x::AbstractArray)
    for layer in net.layers
        x = layer(x)
    end
    return x
end