"""
Represents an optimizer for training a model.
"""
abstract type Optimizer end

"""
Represents the gradient descent optimizer.
- `η` is the learning rate.
"""
struct Descent <: Optimizer
    η::Real
    Descent(η::Real) = new(η)
end

"""
Updates layer using gradient descent optimizer.
- `d` is the optimizer.
- `node` is the node in computational graph.
"""
function train!(d::Descent, node::Union{ScalarOperator, VectorOperator})
    for input in node.inputs
        train!(d, input)
    end
end

train!(::Descent, ::Union{ScalarConstant, MatrixConstant}) = nothing

function train!(d::Descent, node::Union{ScalarVariable, MatrixVariable})
    node.value .-= d.η .* node.∇
end

function onehot(index::Integer, classes::Integer, T::Type{<:Number})
    y = zeros(T, classes)
    y[index + 1] = one(T)
    return y
end

function onecold(y::AbstractArray{T}) where T
    return argmax(y) - 1
end
