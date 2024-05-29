function logsoftmax(x::AbstractArray; dims = 1)
    x = x .- maximum(x; dims = dims)
    return x .- log.(sum(exp.(x); dims = dims))
end

function _validate_sizes(arr1::AbstractArray, arr2::AbstractArray)
    size(arr1) == size(arr2) || throw(DimensionMismatch(
        "loss function expects size(arr1) = $(size(arr1)) to match size(arr2) = $(size(arr2))"
    ))
end

function logitcrossentropy(ŷ, y; dims = 1)
    _validate_sizes(ŷ, y)
    return mean(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))
  end