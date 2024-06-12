using Flux

logsoftmax(y) = y.-log.(sum(exp.(y), dims=1))
logitcrossentropy(ŷ, y) = mean(.-sum(y .* logsoftmax(ŷ); dims=1))
crossentropy(ŷ, y) = -sum(y .* log.(ŷ))

lossfun = crossentropy

x1 = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
x2 = [0.28880380329352523, 0.8055240727553095, 0.7452071295828105]
x3 = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
x4 = [0.28880380329352523, 0.8055240727553095, 0.7452071295828105]

W = [
    0.521213795535383 0.8908786980927811 0.5256623915420473;
    0.5868067574533484 0.19090669902576285 0.3905882754313441
]
Wx = [
    0.5258076696877145 0.917446376762714 0.5454269007849442;
    0.6100961123964701 0.27378680749646933 0.4609900593210261
]
Wh = [
    0.5321058834880117 0.89626945616013;
    0.6501922073422859 0.22227769298279065
]
b = [0.030281630198983913, 0.09937130933301208]

init(out, in) = W[begin:out, begin:in]

rnn = Flux.Recur(Flux.RNNCell(3 => 2; init=init))
rnn1 = Flux.Recur(Flux.RNNCell(tanh, Wx, Wh, b, zeros(2, 1)))

@show ŷ1 = rnn(x1)
@show ŷ2 = rnn(x2)

@show lossfun(ŷ2, ones(2))

Flux.reset!(rnn)
@show Flux.gradient(model -> let
    ŷ1 = rnn(x1)
    ŷ2 = rnn(x2)
    lossfun(ŷ2, ones(2))
end, rnn)

@show ŷ3 = rnn1(x1)
@show ŷ4 = rnn1(x2)

Flux.reset!(rnn1)
@show ŷ5 = rnn1(x3)
@show ŷ6 = rnn1(x4)
