"""
Abstract type that represents layers in a neural network.
"""
abstract type Layer{T} end

"""
Creates a network with the given layers in given order.
- `layers` is a list of layers; each layer should have the same output size as the next 
  layer's input size, otherwise an error is thrown; input size is given in the 'in' field
  and output size is given in the 'out' field of the layer
"""
struct Network{T}
    layers::Base.AbstractVecOrTuple{Layer{T}}

    function Network(layers::Layer{T}...) where T
        nlayers = length(layers)
        for i in 1:nlayers-1
            if hasproperty(layers[i], :out) && hasproperty(layers[i+1], :in)
                in = layers[i].out
                out = layers[i+1].in
                if in != out
                    throw(ArgumentError("Layer $i has output size $in, but layer $(i+1) has input size $out"))
                end
            end
        end
        return new{T}(layers)
    end
end

"""
Applies the network to the input `x`.
- `x` is the input to the network; its size should match the input size of the first layer
  in the network
"""
function (net::Network)(x::AbstractVecOrMat)
    node = MatrixVariable(x)
    for layer in net.layers
        node = layer(node)
    end
    return node
end