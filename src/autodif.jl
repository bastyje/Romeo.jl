module autodif

include("romeo.jl")

# using MLDatasets, Flux
# lossfun = logitcrossentropy

# train_data = MLDatasets.MNIST(split=:train)
# test_data  = MLDatasets.MNIST(split=:test)

# function loader(data; batchsize::Int=1)
#     x1dim = reshape(data.features, 28 * 28, :) # reshape 28×28 pixels into a vector of pixels
#     yhot  = Flux.onehotbatch(data.targets, 0:9) # make a 10×60000 OneHotMatrix
#     Flux.DataLoader((x1dim, yhot); batchsize, shuffle=true)
# end

# net = Chain(
#     RNN((14 * 14) => 64, tanh),
#     Dense(64 => 10, identity),
# )

# using Statistics: mean
# function loss_and_accuracy(model, data)
#     (x,y) = only(loader(data; batchsize=length(data)))
#     Flux.reset!(model)
#     ŷ = model(x[  1:196,:])
#     ŷ = model(x[197:392,:])
#     ŷ = model(x[393:588,:])
#     ŷ = model(x[589:end,:])
#     loss = lossfun(ŷ, y)
#     acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
#     (; loss, acc, split=data.split)
# end

# @show loss_and_accuracy(net, test_data);

# train_log = []
# settings = (;
#     eta = 15e-3,
#     epochs = 5,
#     batchsize = 100,
# )

# opt_state = Flux.setup(Descent(settings.eta), net);

# for epoch in 1:settings.epochs
#     @time for (x,y) in loader(train_data, batchsize=settings.batchsize)
#         Flux.reset!(net)
#         grads = Flux.gradient(model -> let
#                 ŷ = model(x[  1:196,:])
#                 ŷ = model(x[197:392,:])
#                 ŷ = model(x[393:588,:])
#                 ŷ = model(x[589:end,:])
#                 lossfun(ŷ, y)
#             end, net)
#         Flux.update!(opt_state, net, grads[1])
#     end
    
#     loss, acc, _ = loss_and_accuracy(net, train_data)
#     test_loss, test_acc, _ = loss_and_accuracy(net, test_data)
#     @info epoch acc test_acc
#     nt = (; epoch, loss, acc, test_loss, test_acc) 
#     push!(train_log, nt)
# end

# Flux.reset!(net)
# x1, y1 = first(loader(train_data));
# y1hat = net(x1[  1:196,:])
# y1hat = net(x1[197:392,:])
# y1hat = net(x1[393:588,:])
# y1hat = net(x1[589:end,:])
# @show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

# @show loss_and_accuracy(net, train_data);

end # module