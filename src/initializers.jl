"""
Returns matrix of initial weights initalized in glorot uniform manner
"""
function glorot_uniform(::Type{T}, input_size::Integer, output_size::Integer) where T
    scale = sqrt(T(6) / (input_size + output_size))
    return randn(T, input_size, output_size) * scale
end

glorot_uniform(input_size::Integer, output_size::Integer) = glorot_uniform(Float32, input_size, output_size)