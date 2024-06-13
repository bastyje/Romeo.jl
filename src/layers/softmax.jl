"""
Represents a softmax layer.
"""
struct Softmax{T} <: Layer{T}
    Softmax{T}() where T = new{T}()
end

"""
Creates softmax layer computational graph node.
- `x` is the input to the layer
"""
function (layer::Softmax)(x::MatrixNode)
    return _softmax(x)
end

_softmax(x::MatrixNode{T}) where T = VectorOperator(
    (x::MatrixNode{T}) -> _softmax(x.value),
    Function[(x::MatrixNode{T}, g::AbstractVecOrMat{T}) -> _dsoftmax(x.value, g)],
    MatrixNode{T}[x];
    name="softmax"
)

function _softmax(x::AbstractArray{T}) where {T}
    max_ = maximum(x; dims=1)
    if all(isfinite, max_)
        out = exp.(x .- max_)
    else
        _zero, _one, _inf = T(0), T(1), T(Inf)
        out = ifelse(isequal(max_, _inf),
            ifelse(isequal(x, _inf),
                _one,
                _zero),
            exp(x - max_)
        )
    end
    sum!(max_, out)
    out ./= sum!(max_, out)
end

function _dsoftmax(x::AbstractArray{T}, g::AbstractArray{T}) where {T}
    s = _softmax(x)
    out = similar(x, T)
    out .= g .* s .- s .* sum(g .* s; dims=1)
end