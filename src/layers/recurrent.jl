include("network.jl")
include("../graph.jl")

"""
Represents a single vanilla RNN cell.
- `Wx` is the weight matrix for the input with `in` rows and `out` columns
- `Wh` is the weight matrix for the hidden state with `out` rows and `out` columns
- `b` is the bias vector with `out` elements
- `activation` is the activation function to apply to the output,
  should be a function that returns `VectorOperator`
- `state0` is the initial state of the cell
- `in` is the size of the input
- `out` is the size of the output
"""
mutable struct RNNCell{T}
    Wx:: AbstractMatrix{T}
    Wh:: AbstractMatrix{T}
    b:: AbstractVector{T}
    activation:: Function
    state0:: AbstractVector{T}
    in::Integer
    out::Integer

    RNNCell{T}(
        (in, out)::Pair{<:Integer, <:Integer};
        activation::Function=identity,
        init::Function=zeros,
        init_bias::Function=zeros,
        init_state::Function=zeros
    ) where T = new{T}(init(out, in), init(out, out), init_bias(out), activation, init_state(out), in, out)
end

"""
Applies the RNN cell to the input `x` and the state `state`.
- `x` is the input to the cell
- `state` is the state of the cell
"""
function (cell::RNNCell)(x::AbstractArray, state::AbstractArray)
    return cell.activation(cell.Wx * x .+ cell.Wh * state .+ cell.b)
end

"""
Represents recurrent layer with `cell` as the RNN cell.
- `cell` is the RNN cell
- `state` is the state of the RNN cell - initial hidden state
- `in` is the size of the input
- `out` is the size of the output
"""
mutable struct RNN{T} <: Layer{T}
    cell::RNNCell{T}
    state::AbstractVector{AbstractVector{T}}
    in::Integer
    out::Integer
    RNN{T}(cell::RNNCell{T}) where T = new{T}(cell, cell.state0, cell.in, cell.out)
end

"""
Applies the RNN layer to the input `x`.
- `x` is the input to the RNN layer
"""
function (rnn::RNN)(x::AbstractArray)
    state = rnn.cell(x, rnn.state[end])
    push!(rnn.state, state)
    return state
end

# a_t = Wx * x_t + Wh * a_{t-1} + b
# h_t = activation(a_t)
_rnn_step(
    Wx::MatrixNode{T},
    Wh::MatrixNode{T},
    x::MatrixNode{T},
    h::MatrixNode{T},
    b::MatrixNode{T}
) where T = Vector(
    (
        Wx::MatrixNode{T},
        Wh::MatrixNode{T},
        x::MatrixNode{T},
        h::MatrixNode{T},
        b::MatrixNode{T}
     ) -> Wx.value * x.value .+ Wh.value * h.value .+ b.value,
    Function[

        # df/dWx
        (
            Wx::MatrixNode{T},
            Wh::MatrixNode{T},
            x::MatrixNode{T},
            h::MatrixNode{T},
            b::MatrixNode{T},
            g::AbstractVecOrMat{T}
        ) -> g * x.value',

        # df/dx
        (
            Wx::MatrixNode{T},
            Wh::MatrixNode{T},
            x::MatrixNode{T},
            h::MatrixNode{T},
            b::MatrixNode{T},
            g::AbstractVecOrMat{T}
        ) -> Wx.value' * g,
    ],
    MatrixNode{T}[Wx, Wh, x, h, b],
    name="Vanilla RNN cell"
)