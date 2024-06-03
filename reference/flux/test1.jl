using ReverseDiff
using Statistics

logsoftmax(y) = y.-log.(sum(exp.(y), dims=1))
logitcrossentropy(ŷ, y) = mean(.-sum(y .* logsoftmax(ŷ); dims=1))
crossentropy(ŷ, y) = -sum(y .* log.(ŷ))

lossfun = crossentropy

W = [
    0.521213795535383 0.8908786980927811 0.5256623915420473;
    .5868067574533484 0.19090669902576285 0.3905882754313441
]
x = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]

net(W) = lossfun(tanh.(W*x), [1.0, 1.0])

ReverseDiff.gradient(net, W)