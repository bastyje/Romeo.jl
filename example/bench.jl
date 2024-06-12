using Romeo, Flux, Statistics
using ProgressMeter: @showprogress

T = Float32

noisy = rand(T, 2, 1000)
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]
target = Romeo.onehot.(truth, 2, T)
target_decoded = Bool.(Romeo.onecold.(target))

loader = Flux.DataLoader((noisy, target), batchsize=-1, shuffle=true)

init1(::Integer, ::Integer) = Float32[
    1.0885786 0.21650214;
    -0.18102983 0.28195417;
    0.61373 0.33920962
]

init2(::Integer, ::Integer) = Float32[
    0.8970356 -0.5695662 0.24159214;
    -0.8649324 0.5703165 -0.8295718
]

model = Romeo.Network(
    Romeo.Dense{T}(2 => 3, activation=Romeo.tanh, init=init1),
    Romeo.Dense{T}(3 => 2, activation=Romeo.identity, init=init2),
    Romeo.Softmax{T}()
)

out1 = [row for row in eachcol(Romeo.forward!(model(noisy)))]
acc = mean(Romeo.onecold.(out1) .== Romeo.onecold.(target))

@info "Before training" acc

settings = (;
    η = 0.01,
    epochs = 1_000,
    batchsize = 100,
    hi = 2,
    lo = 0.0001
)

optimizer = Romeo.Descent(settings.η)

losses = []
@time @showprogress for epoch in 1:settings.epochs
    for (x, y) in loader
        loss_node = Romeo.crossentropy(model(x), Romeo.MatrixConstant(y))
        Romeo.forward!(loss_node)
        Romeo.backward!(loss_node)
        Romeo.train!(optimizer, loss_node)
        push!(losses, loss_node.value)
    end
end

acc = mean([Romeo.onecold(vec(Romeo.forward!(model(x)))) .== Romeo.onecold(y) for (x, y) in loader])

@info "After training (1)" acc

out2 = Vector{T}[row for row in eachcol(Romeo.forward!(model(noisy)))]
acc = mean(Romeo.onecold.(out2) .== Romeo.onecold.(target))

@info "After training (2)" acc

using Plots

p_true = scatter(noisy[1,:], noisy[2,:], zcolor=target_decoded, title="True classification", legend=false)
p_raw = scatter(noisy[1,:], noisy[2,:], zcolor=out1, title="Before training", legend=false)
p_done = scatter(noisy[1,:], noisy[2,:], zcolor=out2, title="After training", legend=false)

plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))
