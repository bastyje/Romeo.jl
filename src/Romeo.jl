module Romeo

include("graph.jl")
include("operators.jl")

include("layers/network.jl")
include("layers/activation.jl")
include("layers/loss.jl")
include("layers/dense.jl")
include("layers/recurrent.jl")
include("layers/train.jl")

end # module