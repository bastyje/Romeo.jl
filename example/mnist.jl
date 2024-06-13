using Statistics: mean
using ProgressMeter: @showprogress
using Romeo
using LinearAlgebra: norm

# Flux is used only for loading the MNIST dataset
using MLDatasets, Flux

println("Environment: Initialized")

Num = Float32

train_data = MLDatasets.MNIST(split=:train)
test_data = MLDatasets.MNIST(split=:test)

function loader(data; batchsize::Int=-1)
    x1dim = reshape(data.features, 28 * 28, :)
    yhot = Romeo.onehot.(data.targets, 10, Num)
    return Flux.DataLoader((x1dim, yhot); batchsize, shuffle=true)
end

net = Romeo.Network(
    Romeo.RNN(Romeo.RNNCell{Num}(14 * 14 => 64, activation=Romeo.tanh, init=Romeo.glorot_uniform)),
    Romeo.Dense{Num}(64 => 10, activation=Romeo.identity, init=Romeo.glorot_uniform),
    Romeo.Softmax{Num}()
)

function batch_process(model::Romeo.Network, data)
    (x, y) = data

    Romeo.reset!(model)

    ŷ = model(x[  1:196,:])
    ŷ = model(x[197:392,:])
    ŷ = model(x[393:588,:])
    ŷ = model(x[589:end,:])

    return Romeo.crossentropy(ŷ, Romeo.MatrixConstant(y)), ŷ
end

# test the network on entire test data and return accuracy
function loss_and_accuracy(model::Romeo.Network, data)
    (x, y) = only(loader(data; batchsize=length(data)))
    y = hcat(y...)

    loss_node, ŷ = batch_process(model, (x, y))

    loss = Romeo.forward!(loss_node)
    acc = round(100 * mean(Romeo.onecold.(eachcol(ŷ.value)) .== Romeo.onecold.(eachcol(y))); digits=2)

    (; loss, acc, split=data.split)
end

@show loss_and_accuracy(net, test_data)

settings = (;
    η = 1e-2,
    epochs = 5,
    batchsize = 100,
    hi = 2,
    lo = 0.0001
)

optimizer = Romeo.Descent(settings.η)
for epoch in 1:settings.epochs
    @time @showprogress for (x, y) in loader(train_data)
        if length(size(x)) > 1
            y = hcat(y...)
        end
    
        loss_node, _ = batch_process(net, (x, y))

        Romeo.forward!(loss_node)

        if isnan(loss_node.value)
            @error "NaN detected"
            break
        end

        Romeo.backward!(loss_node)
        Romeo.train!(optimizer, loss_node)
    end

    loss, acc, _ = loss_and_accuracy(net, train_data)
    test_loss, test_acc, _ = loss_and_accuracy(net, test_data)
    @info epoch acc, loss, test_acc, test_loss
end
