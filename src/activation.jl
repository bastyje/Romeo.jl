"""
Calculate the softmax of an array of logits.
- `x` is an array of logits.
- `dims` is the dimensions along which the softmax is calculated.
"""
function logsoftmax(x::AbstractArray; dims = 1)
    x = x .- maximum(x; dims = dims)
    return x .- log.(sum(exp.(x); dims = dims))
end

"""
Calculate the cross-entropy loss between the predicted and true labels.
- `ŷ` is an array of predicted labels.
- `y` is an array of true labels.
- `dims` is the dimensions along which the cross-entropy is calculated.
"""
function logitcrossentropy(ŷ, y; dims = 1)
    _validate_sizes(ŷ, y)
    return mean(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))
end

function _validate_sizes(arr1::AbstractArray, arr2::AbstractArray)
    size(arr1) == size(arr2) || throw(DimensionMismatch(
        "loss function expects size(arr1) = $(size(arr1)) to match size(arr2) = $(size(arr2))"
    ))
end