using Statistics: mean

crossentropy(ŷ::MatrixNode{T}, y::MatrixNode{T}) where T = ScalarOperator(
    (ŷ::MatrixNode{T}, y::MatrixNode{T}) -> _crossentropy(ŷ.value, y.value),
    Function[
        # ∂L/∂ŷ
        (ŷ::MatrixNode{T}, y::MatrixNode{T}, g::T) -> -g .* y.value ./ ŷ.value

        # ∂L/∂y
        (ŷ::MatrixNode{T}, y::MatrixNode{T}, g::T) -> nothing
        
        ],
    [ŷ, y],
    name="Crossentropy"
)

function _crossentropy(ŷ::AbstractVecOrMat{T}, y::AbstractVecOrMat{T}; dims = 1, agg = mean, ϵ = eps(T)) where T
    return agg(.-sum(_xlogy.(y, ŷ .+ ϵ); dims = dims))
end

function _xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end