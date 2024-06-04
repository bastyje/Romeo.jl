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
    Wx:: MatrixNode{T}
    Wh:: MatrixNode{T}
    b:: MatrixNode{T}
    activation:: Function
    state0:: MatrixNode{T}
    in::Integer
    out::Integer

    RNNCell{T}(
        (in, out)::Pair{<:Integer, <:Integer};
        activation::Function=identity,
        init::Function=zeros,
        init_bias::Function=zeros,
        init_state::Function=zeros
    ) where T = new{T}(
        MatrixVariable(init(out, in), name="Wx"),
        MatrixVariable(init(out, out), name="Wh"),
        MatrixVariable(init_bias(out), name="b"),
        activation,
        MatrixVariable(init_state(out), name="h"),
        in,
        out
    )
end

"""
Applies the RNN cell to the input `x` and the state `state`.
- `x` is the input to the cell
- `state` is the state of the cell
"""
function (cell::RNNCell)(x::MatrixNode, state::MatrixNode)
    return cell.activation(_rnn_step(cell.Wx, cell.Wh, x, state, cell.b))
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
    state::MatrixNode{T}
    in::Integer
    out::Integer
    RNN(cell::RNNCell{T}) where T = new{T}(cell, cell.state0, cell.in, cell.out)
end

"""
Applies the RNN layer to the input `x`.
- `x` is the input to the RNN layer
"""
function (rnn::RNN)(x::MatrixNode)
    state = rnn.cell(x, rnn.state)
    rnn.state = state
    return state
end

_rnn_step(
    Wx::MatrixNode{T},
    Wh::MatrixNode{T},
    x::MatrixNode{T},
    h::MatrixNode{T},
    b::MatrixNode{T}
) where T = VectorOperator(
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

        # df/dWh
        (
            Wx::MatrixNode{T},
            Wh::MatrixNode{T},
            x::MatrixNode{T},
            h::MatrixNode{T},
            b::MatrixNode{T},
            g::AbstractVecOrMat{T}
        ) -> g * h.value',

        # df/dx
        (
            Wx::MatrixNode{T},
            Wh::MatrixNode{T},
            x::MatrixNode{T},
            h::MatrixNode{T},
            b::MatrixNode{T},
            g::AbstractVecOrMat{T}
        ) -> Wx.value' * g,

        # df/dh
        (
            Wx::MatrixNode{T},
            Wh::MatrixNode{T},
            x::MatrixNode{T},
            h::MatrixNode{T},
            b::MatrixNode{T},
            g::AbstractVecOrMat{T}
        ) -> Wh.value' * g,

        # df/db
        (
            Wx::MatrixNode{T},
            Wh::MatrixNode{T},
            x::MatrixNode{T},
            h::MatrixNode{T},
            b::MatrixNode{T},
            g::AbstractVecOrMat{T}
        ) -> sum(g, dims=2)
    ],
    MatrixNode{T}[Wx, Wh, x, h, b],
    name="Vanilla RNN cell"
)