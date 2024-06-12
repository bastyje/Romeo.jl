# Romeo.jl

Romeo.jl is library aimed for use in training Deep Neural Networks. It is based on concept of computational graph. All networks created in Romeo.jl are embeded into the structure of such graph in order to enable easy computation of gradients inside the network. Gradient computation is based on reverse mode of automatic differentiation (AD).

Romeo.jl enables its users to create their own layers, activation or loss functions and embed them into their custom networks.

There are some examples of running the library in `example` directory.