using Flux, Statistics, ProgressMeter

noisy = rand(Float32, 2, 1000)
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]

model = Chain(
    Dense(2 => 3, tanh),
    Dense(3 => 2),
    softmax)

function dense(x)
    W = randn(2, 3)
    b = randn(3)
    return tanh.(W*x .+ b)
end

out1 = model(noisy)
acc = mean((out1[1,:] .> 0.5) .== truth)
@info "Before training" acc 

target = Flux.onehotbatch(truth, [true, false])
loader = Flux.DataLoader((noisy, target), batchsize=1, shuffle=true);

target = Flux.onehotbatch(truth, [true, false])
loader = Flux.DataLoader((noisy, target), batchsize=-1, shuffle=false);
(x, y) = first(loader)
loss, grads = Flux.withgradient(model) do m
    y_hat = m(x)
    Flux.crossentropy(y_hat, y)
end

@show x
@show y
@show model(x)
Flux.crossentropy(model(x), y)

optim = Flux.setup(Flux.Descent(0.01), model)

losses = []
@showprogress for epoch in 1:1_000
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
end

optim
out2 = model(noisy)

acc = mean((out2[1,:] .> 0.5) .== truth)
@info "After training" acc

using Plots

p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=out1[1,:], title="Untrained network", label="", clims=(0,1))
p_done = scatter(noisy[1,:], noisy[2,:], zcolor=out2[1,:], title="Trained network", legend=false)

plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))