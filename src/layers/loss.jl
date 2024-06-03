crossentropy(ŷ::MatrixNode{T}, y::MatrixNode{T}) where T = ScalarOperator(
    (ŷ::MatrixNode{T}, y::MatrixNode{T}) -> -sum(y.value .* log.(ŷ.value)),
    Function[
        # ∂L/∂ŷ
        (ŷ::MatrixNode{T}, y::MatrixNode{T}, g::Float64) -> -g .* y.value ./ ŷ.value

        # ∂L/∂y
        (ŷ::MatrixNode{T}, y::MatrixNode{T}, g::Float64) -> -g .* log.(ŷ.value)
        ],
    [ŷ, y]
)