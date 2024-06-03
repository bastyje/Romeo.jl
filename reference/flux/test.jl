using Flux

lossfun = Flux.crossentropy

W = [
    0.521213795535383 0.8908786980927811 0.5256623915420473;
    .5868067574533484 0.19090669902576285 0.3905882754313441
]

net = Chain(
    Dense(3 => 2, tanh; init=((::Integer, ::Integer) -> W), bias=false)
)

x = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]

Flux.gradient(model -> lossfun(model(x), [1,1]), net)