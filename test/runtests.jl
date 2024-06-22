using Test
using Romeo
using Test
using Random
using Combinatorics

Random.seed!(1234)

@testset "Graph test" begin
    @testset "Scalars" begin
        x = Romeo.ScalarVariable(5.0)
        two = Romeo.ScalarConstant(2.0)
        squared = x^two
        sine = sin(squared)
        Romeo.forward!(sine)
        Romeo.backward!(sine)
        @test sine.value ≈ -0.13235175009777303
        @test x.∇ ≈ 9.912028118634735
    end

    @testset "Matrices" begin
        A = Romeo.MatrixVariable([1.0 2.0; 3.0 4.0])
        B = Romeo.MatrixConstant([5.0; 6.0])
        C = A * B
        Romeo.forward!(C)
        Romeo.backward!(C)
        @test C.value ≈ [17.0; 39.0]
        @test A.∇ ≈ [5.0 6.0; 5.0 6.0]
    end

    @testset "ScalarConstant" begin
        sc = Romeo.ScalarConstant(5)
        @test Romeo.forward!(sc) == 5
        @test sc.value == 5
    end

    @testset "MatrixConstant" begin
        mc = Romeo.MatrixConstant([1 2; 3 4])
        @test Romeo.forward!(mc) == [1 2; 3 4]
        @test mc.value == [1 2; 3 4]
    end

    @testset "ScalarVariable" begin
        sv = Romeo.ScalarVariable(10)
        @test Romeo.forward!(sv) == 10
        @test sv.value == 10
        sv.value = 20
        @test Romeo.forward!(sv) == 20
        @test sv.value == 20
    end

    @testset "MatrixVariable" begin
        mv = Romeo.MatrixVariable([5 6; 7 8])
        @test Romeo.forward!(mv) == [5 6; 7 8]
        @test mv.value == [5 6; 7 8]
        mv.value = [9 10; 11 12]
        @test Romeo.forward!(mv) == [9 10; 11 12]
        @test mv.value == [9 10; 11 12]
    end

    @testset "ScalarOperator" begin
        so = Romeo.ScalarOperator(
            (x::Romeo.ScalarNode, y::Romeo.ScalarNode) -> x.value + y.value,
            Function[(x::Romeo.ScalarNode, y::Romeo.ScalarNode, g::Int64) -> 1, (x::Romeo.ScalarNode, y::Romeo.ScalarNode, g::Int64) -> 1],
            Romeo.ScalarNode{Int64}[Romeo.ScalarConstant(1), Romeo.ScalarConstant(2)]
        )
        @test Romeo.forward!(so) == 3
        @test so.value == 3
        
        so = Romeo.ScalarOperator(
            (x::Romeo.ScalarNode, y::Romeo.ScalarNode) -> x.value * y.value,
            Function[(x::Romeo.ScalarNode, y::Romeo.ScalarNode, g::Int64) -> y.value, (x::Romeo.ScalarNode, y::Romeo.ScalarNode, g::Int64) -> x.value],
            Romeo.ScalarNode{Int64}[Romeo.ScalarConstant(2), Romeo.ScalarConstant(3)]
        )
        @test Romeo.forward!(so) == 6
        @test so.value == 6
    end

    @testset "VectorOperator" begin
        vo = Romeo.VectorOperator(
            (x::Romeo.MatrixNode, y::Romeo.MatrixNode) -> x.value + y.value,
            Function[(x::Romeo.MatrixNode, y::Romeo.MatrixNode, g::AbstractVecOrMat{Int64}) -> g, (x::Romeo.MatrixNode, y::Romeo.MatrixNode, g::AbstractVecOrMat{Int64}) -> g],
            Romeo.MatrixNode{Int64}[Romeo.MatrixConstant([1 2; 3 4]), Romeo.MatrixConstant([5 6; 7 8])]
        )
        @test Romeo.forward!(vo) == [6 8; 10 12]
        @test vo.value == [6 8; 10 12]
        
        vo = Romeo.VectorOperator(
            (x::Romeo.MatrixNode, y::Romeo.MatrixNode) -> x.value * y.value,
            Function[(x::Romeo.MatrixNode, y::Romeo.MatrixNode, g::AbstractVecOrMat{Int64}) -> g * y.value', (x::Romeo.MatrixNode, y::Romeo.MatrixNode, g::AbstractVecOrMat{Int64}) -> x.value' * g],
            Romeo.MatrixNode{Int64}[Romeo.MatrixConstant([1 2; 3 4]), Romeo.MatrixConstant([5 6; 7 8])]
        )
        @test Romeo.forward!(vo) == [19 22; 43 50]
        @test vo.value == [19 22; 43 50]
    end

    @testset "ScalarOperator backward pass" begin
        so = Romeo.ScalarOperator(
            (x::Romeo.ScalarNode, y::Romeo.ScalarNode) -> x.value * y.value,
            Function[(x::Romeo.ScalarNode, y::Romeo.ScalarNode, g::Int64) -> y.value, (x::Romeo.ScalarNode, y::Romeo.ScalarNode, g::Int64) -> x.value],
            Romeo.ScalarNode{Int64}[Romeo.ScalarVariable(1), Romeo.ScalarConstant(2)]
        )
        Romeo.forward!(so)
        Romeo.backward!(so)
        @test so.∇ == 1
        @test so.inputs[1].∇ == 2
    end

    @testset "VectorOperator backward pass" begin
        vo = Romeo.VectorOperator(
            (x::Romeo.MatrixNode, y::Romeo.MatrixNode) -> x.value * y.value,
            Function[(x::Romeo.MatrixNode, y::Romeo.MatrixNode, g::AbstractVecOrMat{Int64}) -> g * y.value', (x::Romeo.MatrixNode, y::Romeo.MatrixNode, g::AbstractVecOrMat{Int64}) -> x.value' * g],
            Romeo.MatrixNode{Int64}[Romeo.MatrixVariable([1 2; 3 4]), Romeo.MatrixConstant([5; 6])]
        )
        Romeo.forward!(vo)
        Romeo.backward!(vo)
        @test vo.∇ == ones(size(vo.value))
        @test vo.inputs[1].∇ == [5 6; 5 6]
    end
end

@testset "Layers tests" begin
    @testset "Network" begin
        @testset "Network with two Dense layers with non-trainable biases" begin
            x = [1.0; 2.0; 3.0]
            W1 = [
                0.5731809195726594 0.6648371921346286 1.2993364821337263;
                1.7673004203595648 2.3973935493382115 0.9201088691339511;
                1.3454095474399999 0.7717639140663247 0.4680042029118444;
                0.9759970753615571 0.12931582541678244 2.0914188568559973
            ]
            dense1 = Romeo.Dense{Float64}(
                3 => 4,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W1)
            )
            W2 = [
                1.335362103686751 0.9609312644677062 1.3767399069930406 0.629081172867276;
                0.45009484124196886 0.6335767397124203 0.43450802801331634 0.2674203454904461
            ]
            dense2 = Romeo.Dense{Float64}(
                4 => 2,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W2)
            )
            net = Romeo.Network(dense1, dense2)
            ŷ = net(x)
            loss = Romeo.crossentropy(ŷ, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)
            Romeo.backward!(loss)

            @test loss.value ≈ 0.05664527957429444
            @test dense1.W.∇ ≈ [
                -1.8910701026126866e-6 -3.7821402052253732e-6 -5.67321030783806e-6;
                -2.303485446852047e-9 -4.606970893704094e-9 -6.910456340556141e-9;
                -3.728764927217531e-5 -7.457529854435061e-5 -0.00011186294781652593;
                -3.675598583059508e-8 -7.351197166119016e-8 -1.1026795749178525e-7
            ]
        end

        @testset "Simple network with one dense layer train, fired twice more after training" begin
            x = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
            W = [
                0.521213795535383 0.8908786980927811 0.5256623915420473;
                0.5868067574533484 0.19090669902576285 0.3905882754313441
            ]
            Wcopy = copy(W)
            dense = Romeo.Dense{Float64}(
                3 => 2,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W),
                bias=true
            )
            net = Romeo.Network(dense)

            ŷ = net(x)
            loss = Romeo.crossentropy(ŷ, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)
            Romeo.backward!(loss)

            # calculated using ReverseDiff:
            # -----------------------------
            # using ReverseDiff
            #
            # crossentropy(ŷ, y) = -sum(y .* log.(ŷ))
            # lossfun = crossentropy
            # W = [
            #     0.521213795535383 0.8908786980927811 0.5256623915420473;
            #     0.5868067574533484 0.19090669902576285 0.3905882754313441
            # ]
            # x = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
            # b = zeros(2)
            # netW(W) = lossfun(tanh.(W*x+b), [1.0, 1.0])
            # netb(b) = lossfun(tanh.(W*x+b), [1.0, 1.0])
            # @show ReverseDiff.gradient(netW, W)
            # @show ReverseDiff.gradient(netb, b)

            @test loss.value ≈ 1.097683786749982
            @test dense.W.∇ ≈ [
                -0.017787954879484345 -0.3704414365214419 -0.23042022415741797;
                -0.09207931452273083 -1.9175893899444665 -1.192769851078242
            ]
            @test dense.b.∇ ≈ [-0.3968930538640049, -2.054516136690936]

            # update weights
            η = 0.1
            optimizer = Romeo.Descent(η)
            Romeo.train!(optimizer, loss)

            @test dense.W.value ≈ Wcopy .- η .* dense.W.∇
            @test dense.b.value ≈ zeros(size(dense.b.value)) .- η .* dense.b.∇

            # forward pass after training
            ŷ = net(ones(3))
            loss = Romeo.crossentropy(ŷ, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)

            @test loss.value ≈ 0.1014433297795401

            # another forward pass after training
            x = [0.28880380329352523, 0.8055240727553095, 0.7452071295828105]
            ŷ = net(x)
            loss = Romeo.crossentropy(ŷ, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)

            @test loss.value ≈ 0.3738259704545526
        end

        @testset "Network with recurrent later, trained and then executed" begin
            x1 = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
            x2 = [0.28880380329352523, 0.8055240727553095, 0.7452071295828105]
            W = [
                0.521213795535383 0.8908786980927811 0.5256623915420473;
                0.5868067574533484 0.19090669902576285 0.3905882754313441
            ]

            init(::Type, out, in) = W[begin:out, begin:in]
            
            Wxcopy = init(Float32, 2, 3)
            Whcopy = init(Float32, 2, 2)

            rnn = Romeo.RNN(Romeo.RNNCell{Float64}(
                3 => 2,
                activation = Romeo.tanh,
                init = init
            ))

            net = Romeo.Network(rnn)

            ŷ1 = net(x1)
            ŷ2 = net(x2)

            loss = Romeo.crossentropy(ŷ2, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)
            Romeo.backward!(loss)

            @test loss.value ≈ 0.2249806351451192
            @test ŷ1.value ≈ [0.8210538558264054, 0.40635943187979984]
            @test ŷ2.value ≈ [0.9673848437916395, 0.8254539958257284]

            # update weights
            η = 0.1
            optimizer = Romeo.Descent(η)
            Romeo.train!(optimizer, loss)

            @test rnn.cell.Wx.value ≈ [
                0.5258076696877145 0.917446376762714 0.5454269007849442;
                0.6100961123964701 0.27378680749646933 0.4609900593210261
            ]
            @test rnn.cell.Wh.value ≈ [
                0.5321058834880117 0.89626945616013;
                0.6501922073422859 0.22227769298279065
            ]
            @test rnn.cell.b.value ≈ [0.030281630198983913, 0.09937130933301208]

            Romeo.reset!(net)
            
            @test rnn.state.value ≈ zeros(2)
            
            x1 = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
            x2 = [0.28880380329352523, 0.8055240727553095, 0.7452071295828105]

            # forward pass after training
            ŷ3 = net(x1)
            ŷ4 = net(x2)

            loss = Romeo.crossentropy(ŷ4, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)

            @test loss.value ≈ 0.11751629453773474
            @test ŷ3.value ≈ [0.8416491825151537, 0.5715937537923143]
            @test ŷ4.value ≈ [0.9796084367053993, 0.9076341019695154]

            Romeo.reset!(net)

            # another forward pass after training
            x1 = [0.28880380329352523, 0.8055240727553095, 0.7452071295828105]
            x2 = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
            ŷ5 = net(x1)
            ŷ6 = net(x2)

            loss = Romeo.crossentropy(ŷ6, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)

            @test loss.value ≈ 0.15004204177158917
            @test ŷ5.value ≈ [0.8686659625276374, 0.6856206300362764]
            @test ŷ6.value ≈ [0.9802348414670768, 0.8780261168940096]

        end

        @testset "Network with three dense layers with trainable biases" begin
            x = [1.0; 2.0; 3.0; 4.0; 5.0]
            W1 = [
                0.5731809195726594 0.6648371921346286 1.2993364821337263 1.7673004203595648 2.3973935493382115;
                1.3454095474399999 0.7717639140663247 0.4680042029118444 0.9759970753615571 0.12931582541678244;
                1.335362103686751 0.9609312644677062 1.3767399069930406 0.629081172867276 0.45009484124196886;
                0.6335767397124203 0.43450802801331634 0.2674203454904461 0.5731809195726594 0.6648371921346286
            ]
            dense1 = Romeo.Dense{Float64}(
                5 => 4,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W1),
                bias=true
            )
            W2 = [
                0.5731809195726594 0.6648371921346286 1.2993364821337263 1.7673004203595648;
                1.7673004203595648 2.3973935493382115 0.9201088691339511 1.3454095474399999;
                1.3454095474399999 0.7717639140663247 0.4680042029118444 0.9759970753615571
            ]
            dense2 = Romeo.Dense{Float64}(
                4 => 3,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W2),
                bias=true
            )
            W3 = [
                0.5731809195726594 0.6648371921346286 1.2993364821337263;
                1.7673004203595648 2.3973935493382115 0.9201088691339511
            ]
            dense3 = Romeo.Dense{Float64}(
                3 => 2,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W3),
                bias=true
            )
            net = Romeo.Network(dense1, dense2, dense3)

            ŷ = net(x)
            loss = Romeo.crossentropy(ŷ, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)
            Romeo.backward!(loss)

            @test loss.value ≈ 0.01264078864322686
            @test dense1.W.∇ ≈ [
                0.0 0.0 0.0 0.0 0.0;
                -7.421688063878206e-12 -1.4843376127756412e-11 -2.2265064191634617e-11 -2.9686752255512823e-11 -3.710844031939103e-11;
                -7.0459323714590874e-15 -1.4091864742918175e-14 -2.1137797114377262e-14 -2.818372948583635e-14 -3.522966185729544e-14;
                -6.436749253428762e-11 -1.2873498506857524e-10 -1.9310247760286287e-10 -2.574699701371505e-10 -3.218374626714381e-10
            ]

            # TODO add remaining tests
        end
    end

    @testset "Dense layer as graph node" begin
        @testset "Small weight matrix with non-trainable bias" begin
            W = [
                0.521213795535383 0.8908786980927811 0.5256623915420473;
                .5868067574533484 0.19090669902576285 0.3905882754313441
            ]
            dense = Romeo.Dense{Float64}(
                3 => 2,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W),
                bias=false
            )
            x = Romeo.MatrixConstant([0.044818005017491114, 0.933353287277165, 0.5805599818745412])
            ŷ = dense(x)
            loss = Romeo.crossentropy(ŷ, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)
            Romeo.backward!(loss)

            @test loss.value ≈ 1.097683786749982

            # calculated using ReverseDiff:
            # -----------------------------
            # using ReverseDiff
            #
            # crossentropy(ŷ, y) = -sum(y .* log.(ŷ))
            # lossfun = crossentropy
            # W = [
            #     0.521213795535383 0.8908786980927811 0.5256623915420473;
            #     .5868067574533484 0.19090669902576285 0.3905882754313441
            # ]
            # x = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
            # net(W) = lossfun(tanh.(W*x), [1.0, 1.0])
            # @show ReverseDiff.gradient(net, W)

            @test dense.W.∇ ≈ [
                -0.017787954879484345 -0.3704414365214419 -0.23042022415741797;
                -0.09207931452273083 -1.9175893899444665 -1.192769851078242
            ]
            @test !hasproperty(dense.b, :∇)
        end

        @testset "Big weight matrix with non-trainable bias" begin
            W = [
                0.9032294842032307 1.4580230482780379 0.9310737880870507 1.1078463467814603 0.9148034098230938;
                0.11668451330809364 0.22821957601479792 1.7219137174748527 0.50884994302868 0.46319956459103007;
                1.8935648128511766 0.9149075179410003 0.5904856093147604 1.1312243565957947 0.7803309323918235;
                0.49135525694170146 0.3300399360866784 0.22673982804736076 1.590391961842889 0.5721860571075313;
                0.891552228329739 0.7915383581003806 0.5371599284764134 0.22198555796012634 0.23241707056348793;
                0.22886890334681814 0.21341787856291763 0.7490686531318762 0.7185754431704935 0.8726286623779024;
                1.1562322112758119 2.109483277071983 1.209482561668428 1.9545604047295333 0.5242536937822103;
                1.5155150335659924 0.9407211049689689 0.9224087705945953 1.2013672245406082 0.31311150641510643;
                0.9213104293760906 0.0299181242563318 0.731878698017706 1.021678835876467 0.9093530941420519;
                0.21335406646523716 1.064224607244219 0.7303425396061004 0.1751511905658525 1.3936092070796675;
                0.9468201683854216 1.6914066026373635 0.20533720641846576 0.25583214386307485 1.5072767964876896;
                1.111122722720771 0.6070038304951009 0.2544244415583391 1.1849201012991477 0.32010980719644705;
                0.4575974148040482 0.1640054823968387 0.46241113281414264 0.2875765220950689 0.283116690902816;
                1.687603194498139 0.8457670485226456 2.1167917563888663 1.122888164838142 0.5292648475973878;
                0.5124010448525874 1.7656762271568327 0.1833707842698142 0.39038171637593544 1.2530040989828652;
                1.5434777877637214 2.5877598854408874 0.292590332753358 0.2661670132881214 0.5562683547179444;
                0.4939106000569548 0.06421637865406256 1.2457654742036028 1.2014087629756702 1.3108846702846528;
                0.3687249423335804 2.2461988122693843 0.8765342199675517 0.6416327045366876 0.5301017038940709;
                0.07209062200390745 1.8281939078892828 0.5151602322277371 2.900678772089583 1.1550754460222896;
                0.6988334283674431 0.5786021382871562 0.6786387647135722 2.410415564128154 2.0579203641852986
            ]
            dense = Romeo.Dense{Float64}(
                5 => 20,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W),
                bias=false
            )
            x = Romeo.MatrixConstant([
                0.044818005017491114,
                0.933353287277165,
                0.5805599818745412, 
                0.5468807556603925,
                0.05129481188265768
            ])
            ŷ = dense(x)
            loss = Romeo.crossentropy(ŷ, Romeo.MatrixConstant(ones(20)))

            Romeo.forward!(loss)
            Romeo.backward!(loss)

            @test loss.value ≈ 1.94094645094747

            # calculated using ReverseDiff:
            # -----------------------------
            # using ReverseDiff
            #
            # crossentropy(ŷ, y) = -sum(y .* log.(ŷ))
            # lossfun = crossentropy
            # W = [
            #     0.9032294842032307 1.4580230482780379 0.9310737880870507 1.1078463467814603 0.9148034098230938;
            #     0.11668451330809364 0.22821957601479792 1.7219137174748527 0.50884994302868 0.46319956459103007;
            #     1.8935648128511766 0.9149075179410003 0.5904856093147604 1.1312243565957947 0.7803309323918235;
            #     0.49135525694170146 0.3300399360866784 0.22673982804736076 1.590391961842889 0.5721860571075313;
            #     0.891552228329739 0.7915383581003806 0.5371599284764134 0.22198555796012634 0.23241707056348793;
            #     0.22886890334681814 0.21341787856291763 0.7490686531318762 0.7185754431704935 0.8726286623779024;
            #     1.1562322112758119 2.109483277071983 1.209482561668428 1.9545604047295333 0.5242536937822103;
            #     1.5155150335659924 0.9407211049689689 0.9224087705945953 1.2013672245406082 0.31311150641510643;
            #     0.9213104293760906 0.0299181242563318 0.731878698017706 1.021678835876467 0.9093530941420519;
            #     0.21335406646523716 1.064224607244219 0.7303425396061004 0.1751511905658525 1.3936092070796675;
            #     0.9468201683854216 1.6914066026373635 0.20533720641846576 0.25583214386307485 1.5072767964876896;
            #     1.111122722720771 0.6070038304951009 0.2544244415583391 1.1849201012991477 0.32010980719644705;
            #     0.4575974148040482 0.1640054823968387 0.46241113281414264 0.2875765220950689 0.283116690902816;
            #     1.687603194498139 0.8457670485226456 2.1167917563888663 1.122888164838142 0.5292648475973878;
            #     0.5124010448525874 1.7656762271568327 0.1833707842698142 0.39038171637593544 1.2530040989828652;
            #     1.5434777877637214 2.5877598854408874 0.292590332753358 0.2661670132881214 0.5562683547179444;
            #     0.4939106000569548 0.06421637865406256 1.2457654742036028 1.2014087629756702 1.3108846702846528;
            #     0.3687249423335804 2.2461988122693843 0.8765342199675517 0.6416327045366876 0.5301017038940709;
            #     0.07209062200390745 1.8281939078892828 0.5151602322277371 2.900678772089583 1.1550754460222896;
            #     0.6988334283674431 0.5786021382871562 0.6786387647135722 2.410415564128154 2.0579203641852986
            # ]
            # x = [ 0.044818005017491114, 0.933353287277165, 0.5805599818745412, 0.5468807556603925, 0.05129481188265768 ]
            # net(W) = lossfun(tanh.(W*x), ones(20))
            # @show ReverseDiff.gradient(net, W)

            @test dense.W.∇ ≈ [
                -0.00099961498916122 -0.02081739104141367 -0.012948734772162877 -0.012197557665241774 -0.001144072852062437;
                -0.008595932127595797 -0.1790138027199511 -0.11134931593327678 -0.10488976150791175 -0.009838160383735051;
                -0.0037016059388889386 -0.07708745781786411 -0.04794957463968699 -0.04516794204085037 -0.004236537976755878;
                -0.01184168999335978 -0.246608037950507 -0.15339396131590102 -0.14449532881566954 -0.013552974085863109;
                -0.01562037768938411 -0.3253007548909791 -0.20234203161615827 -0.19060384216426512 -0.01787773317443111;
                -0.020864467777323484 -0.4345110760653732 -0.27027251725946977 -0.25459357015247447 -0.023879665086652093;
                -8.642467324264827e-5 -0.0017998291722578815 -0.001119521199385902 -0.0010545759587513163 -9.891420544648701e-5; 
                -0.002410976091591005 -0.0502095633162363 -0.031231114269538753 -0.029419346674039854 -0.0027593946902250777;
                -0.02013157759120279 -0.4192483381509577 -0.26077886146189005 -0.2456506567263782 -0.023040862373032076;
                -0.007406282896195656 -0.15423887085047736 -0.09593893039850827 -0.09037335743358925 -0.008476590775556302;
                -0.0035758449333303717 -0.07446843343461544 -0.046320501523221605 -0.0436333740982202 -0.004092602807848136;
                -0.010331913618202461 -0.21516632736440358 -0.13383673772565272 -0.1260726514876178 -0.011825014639244465;
                -0.05745378554305267 -1.1964985853832044 -0.7442403712633091 -0.7010657801718831 -0.06575663330456784;
                -0.0007546773336599011 -0.015716464173943845 -0.009775880452057788 -0.00920876577060102 -0.0008637383981523176;
                -0.002941145604441132 -0.061250560286977854 -0.0380987828025442 -0.035888610616119046 -0.003366180856118766;
                -0.0006262872591877423 -0.013042688355149991 -0.008112751107515623 -0.007642117256923852 -0.0007167942243748464;
                -0.008431229141826367 -0.175583795669609 -0.10921579922730859 -0.10288001356661101 -0.009649655592674899;
                -0.0004443813743528325 -0.009254423894929963 -0.005756392827815036 -0.005422455143030932 -0.0005086004830578566;
                -0.000120083118698674 -0.002500780066853934 -0.0015555233479922347 -0.0014652849155235264 -0.00013743675073282297;
                -0.0015079210906167356 -0.031403073526642324 -0.019533190749901366 -0.018400038671754496 -0.0017258360484115852
            ]
            @test !hasproperty(dense.b, :∇)
        end

        @testset "Small weight matrix with trainable bias" begin
            W = [
                0.521213795535383 0.8908786980927811 0.5256623915420473;
                0.5868067574533484 0.19090669902576285 0.3905882754313441
            ]
            dense = Romeo.Dense{Float64}(
                3 => 2,
                activation = Romeo.tanh,
                init = ((::Type, ::Integer, ::Integer) -> W),
                bias=true
            )
            x = Romeo.MatrixConstant([0.044818005017491114, 0.933353287277165, 0.5805599818745412])
            ŷ = dense(x)
            loss = Romeo.crossentropy(ŷ, Romeo.MatrixConstant(ones(2)))

            Romeo.forward!(loss)
            Romeo.backward!(loss)

            # calculated using ReverseDiff:
            # -----------------------------
            # using ReverseDiff
            #
            # crossentropy(ŷ, y) = -sum(y .* log.(ŷ))
            # lossfun = crossentropy
            # W = [
            #     0.521213795535383 0.8908786980927811 0.5256623915420473;
            #     0.5868067574533484 0.19090669902576285 0.3905882754313441
            # ]
            # x = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
            # b = zeros(2)
            # netW(W) = lossfun(tanh.(W*x+b), [1.0, 1.0])
            # netb(b) = lossfun(tanh.(W*x+b), [1.0, 1.0])
            # @show ReverseDiff.gradient(netW, W)
            # @show ReverseDiff.gradient(netb, b)

            @test loss.value ≈ 1.097683786749982
            @test dense.W.∇ ≈ [
                -0.017787954879484345 -0.3704414365214419 -0.23042022415741797;
                -0.09207931452273083 -1.9175893899444665 -1.192769851078242
            ]
            @test dense.b.∇ ≈ [-0.3968930538640049, -2.054516136690936]
        end
    end

    @testset "Recurrent layer as graph node" begin
        W = [
            0.521213795535383 0.8908786980927811 0.5256623915420473;
            0.5868067574533484 0.19090669902576285 0.3905882754313441
        ]
        init(::Type, out, in) = W[begin:out, begin:in]
        rnn = Romeo.RNN(Romeo.RNNCell{Float64}(
            3 => 2,
            activation = Romeo.tanh,
            init = init
        ))

        x1 = Romeo.MatrixConstant([0.044818005017491114, 0.933353287277165, 0.5805599818745412])
        x2 = Romeo.MatrixConstant([0.5468807556603925, 0.05129481188265768, 0.28880380329352523])

        ŷ1 = rnn(x1)
        ŷ2 = rnn(x2)
        loss = Romeo.crossentropy(ŷ2, Romeo.MatrixConstant(ones(2)))

        Romeo.forward!(loss)
        a = 1

        # calculated using Flux:
        # ----------------------
        # using Flux

        # W = [
        #     0.521213795535383 0.8908786980927811 0.5256623915420473;
        #     0.5868067574533484 0.19090669902576285 0.3905882754313441
        # ]
        # init(out, in) = W[begin:out, begin:in]
        # rnn = Flux.Recur(Flux.RNNCell(3 => 2; init=init))
        # x1 = [0.044818005017491114, 0.933353287277165, 0.5805599818745412]
        # x2 = [0.5468807556603925, 0.05129481188265768, 0.28880380329352523]
        # @show ŷ1 = rnn(x1)
        # @show ŷ2 = rnn(x2)
        #
        # crossentropy(ŷ, y) = -sum(y .* log.(ŷ))
        # lossfun = crossentropy
        #
        # Flux.reset!(rnn)
        # Flux.gradient(model -> let
        #     ŷ = model(x1)
        #     ŷ = model(x2)
        #     lossfun(ŷ, y)
        # end, rnn)

        @test loss.value ≈ 0.42801981584327553 # TODO 0.4280197933495288 ?
        @test ŷ1.value ≈ [0.8210538558264054, 0.40635943187979984]
        @test ŷ2.value ≈ [0.8544775959198981, 0.7628035146943225]
        
        Romeo.backward!(loss)
        a = 1

        @test rnn.cell.Wx.∇ ≈ [
            -0.17982226168319812 -0.16410117139911734 -0.1832089327968354;
            -0.31421619307927734 -0.3289080896243073 -0.3454043194471258
        ]
        @test rnn.cell.Wh.∇ ≈ [
            -0.2593118942761356 -0.1283397354215488;
            -0.45006074901684556 -0.22274596240054437
        ] 
        @test rnn.cell.b.∇ ≈ [-0.47428990391611636, -0.8704191002157344]
        
        # currently nvm
        # @test rnn.state.∇ ≈ [-0.27170208, -0.20269352]
    end
end

include("../example/mnist.jl")