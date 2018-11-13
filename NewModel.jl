using Knet
using Distributions

function initdatadaily(;trnlength = 3000, devtstlength = 200)
    inputs = Array{Float32,2}(readdlm("inputs.txt"))
    specs = Array{String}(readdlm("specs.txt"))[:]
    ygolds = Array{Float32}(readdlm("ygolds.txt"))[:]
    x, y, mean, stddev = datapreprocess(inputs, specs, ygolds, trnlength)
    return x[:,1:trnlength + devtstlength], y[1:trnlength + devtstlength], mean, stddev
end

function datapreprocess(inputs, specs, ygolds, trnlength)
    transformedinputs = logtransfomer(inputs, specs)
    transformedygolds = log.(ygolds)
    inputsbeforenorm = subtractor(transformedinputs)
    ygoldsbeforenorm = subtractor(transformedygolds)
    total = vcat(inputs[:,2:end], inputsbeforenorm)
    x, y, mean, stddev = standardization(total, ygoldsbeforenorm, trnlength)
    return x, y, mean, stddev
end

function standardization(data, ygolds, trnlength)
    staty = ygolds[1:trnlength - 1]
    statdata = data[:,2:trnlength]
    meandata = mean(statdata, 2)
    meany = mean(staty)
    stddata = std(statdata, 2, corrected = false)
    stdy = std(staty, corrected = false)
    (data .- meandata) ./ stddata, (ygolds .- meany) ./ stdy, meany, stdy
end

function subtractor(data)
    if ndims(data) == 1
        yt1 = data[2:end]
        yt0 = data[1:end-1]
        return yt1 .- yt0
    end
    if ndims(data) == 2
        yt1 = data[:,2:end]
        yt0 = data[:,1:end-1]
        return yt1 .- yt0
    end
end

function logtransfomer(inputs, specs)
    logtransform = Array{Bool}(size(specs))
    logtransform[specs .== "level"] = true
    logtransform[specs .== "percent"] = false
    outputs = copy(inputs)
    outputs[logtransform,:] = log.(inputs[logtransform,:])
    return outputs
end

function minibatcher(inputs, ygold, batchlength, trainlength, devlength, testlength)
    len = size(inputs, 2)
    inputdim = size(inputs, 1)
    trn = []
    dev = []
    tst = []
    for i = 1 :batchlength:trainlength
        bl = min(i + batchlength -1, trainlength)
        input = inputs[:,[i:bl...]]
        input = reshape(input, inputdim, 1, :)
        ygol = ygold[i:bl]
        if gpu() != -1
            input = convert(KnetArray,input)
            ygol = convert(KnetArray, ygol)
        end
        push!(trn, (input,ygol))
    end
    for i = trainlength + 1:batchlength: trainlength + devlength
        bl = min(i + batchlength -1, trainlength + devlength)
        input = inputs[:,[i:bl...]]
        input = reshape(input, inputdim, 1, :)
        ygol = ygold[i:bl]
        if gpu() != -1
            input = convert(KnetArray,input)
            ygol = convert(KnetArray, ygol)
        end
        push!(dev, (input,ygol))
    end
    for i = trainlength + devlength + 1 :batchlength: trainlength + devlength + testlength
        bl = min(i + batchlength -1, trainlength + devlength + testlength)
        input = inputs[:,[i:bl...]]
        input = reshape(input, inputdim, 1, :)
        ygol = ygold[i:bl]
        if gpu() != -1
            input = convert(KnetArray, input)
            ygol = convert(KnetArray, ygol)
        end
        push!(tst, (input,ygol))
    end
    return trn, dev, tst
end

function weightsinit(inputs, hiddensize, layernum, rnntype, dropouthidden)
    winit = 0.1
    inputsize =  size(inputs, 1)
    r, wr = rnninit(inputsize, hiddensize, rnnType = rnntype, numLayers = layernum, dropout = dropouthidden)
    wy = gpu() == -1 ? Array{Float32,2}(xavier(1,hiddensize) * winit) : KnetArray{Float32,2}(xavier(1,hiddensize) * winit)
    by = gpu() == -1 ? Array{Float32,2}(zeros(1, 1)) : KnetArray{Float32,2}(zeros(1, 1))
    return r, wr, wy, by
end

function hiddeninit(w)
    h = Any[]
    hiddensize = w[1].hiddenSize
    numlayers = w[1].numLayers
    rnntype = w[1].mode
    inithiddens = zeros(Float32, hiddensize, 1, numlayers)
    initcells = zeros(Float32, hiddensize, 1, numlayers)
    if gpu() != -1
        inithiddens = convert(KnetArray{Float32}, inithiddens)
        initcells = convert(KnetArray{Float32}, initcells)
    end
    push!(h, inithiddens)
    if rnntype == 2
        push!(h, initcells)
        return h
    end
    if rnntype != 2
        push!(h, nothing)
        return h
    end
end

function predict(w, input, h; gaussianinput = 0, dropoutinput = 0, dropouthidden = 0)
    hx, cx = h
    r, wr, wy, by = w
    inputnoisevector = gpu() == -1 ? Array{Float32}(gaussian(size(input), mean = 1, std = gaussianinput)) : KnetArray{Float32}(gaussian(size(input), mean = 1, std = gaussianinput))
    input = input .* inputnoisevector
    input = dropout(input, dropoutinput)
    hiddens, hx, cx, = rnnforw(r, wr, input, hx, cx)
    h[1] = hx
    h[2] = cx
    hiddens = reshape(hiddens, size(hiddens,1), :)
    hiddens = dropout(hiddens, dropouthidden)
    yhat = wy * hiddens .+ by
    return yhat[:]
end

function loss(w, data, h, gaussianx, dropoutx, dropouth)
    input, ygold = data
    yhat = predict(w, input, h)#, gaussianinput = gaussianx, dropoutinput = dropoutx, dropouthidden = dropouth)
    1/2 * mean(abs2, yhat - ygold)
end

lossgrad = grad(loss)

function train(w, batches, h, gaussianinput, dropoutinput, dropouthidden)
    optim = optimizers(w, Adadelta)
    for data in batches
        gradamount = lossgrad(w, data, h, gaussianinput, dropoutinput, dropouthidden)
        update!(w, gradamount, optim)
    end
end

function reversestandardization(input, means, stddevs)
    input * stddevs .+ means
end

function predictiongenerator(w, batcheddata, means, stddevs)
    h = hiddeninit(w)
    yhats = Any[]
    ygolds = Any[]
    for data in batcheddata
        input, ygold = data
        yhat = predict(w, input, h)
        yhat = exp.(reversestandardization(yhat, means, stddevs))
        ygold = exp.(reversestandardization(ygold, means, stddevs))
        push!(yhats, yhat)
        push!(ygolds, ygold)
    end
    return vcat(yhats...), vcat(ygolds...)
end

function coefficientofdetermination(ygolds, yhats)
    ymean = mean(ygolds)
    SSRes = sum((ygolds .- yhats).^2)
    SSTotal = sum((ygolds .- ymean).^2)
    1 - SSRes/SSTotal
end

function returncalc(ygolds, yhats; leverage = 1, commissions = 0 / 10000)
    yacts = copy(yhats)
    yacts[yacts .< 1] = -1
    yacts[yacts .> 1] = 1
    marketchange = (ygolds .- 1) * leverage
    realized = yacts .* marketchange
    accseries = copy(realized)
    accseries[accseries .< 0] = 0
    accseries[accseries .> 0] = 1
    prodreturnyearly = prod(realized .+ 1)^(261/length(realized)) * (1 - 2 * commissions * leverage)^261
    return prodreturnyearly, mean(accseries)
end

function dieboldmariano(ygolds, yhats)
    modelerrors = ygolds .- yhats
    randomerrors = ygolds .- 1
    modellosses = modelerrors .^2 * 1/2
    randomlosses = randomerrors .^2 * 1/2
    lossdiffs = randomlosses - modellosses
    dbar = mean(lossdiffs)
    t = length(lossdiffs)
    gamma = mean((lossdiffs .- dbar).^2)
    stat = dbar / ((gamma/t)^(1/2))
    return stat
end

function report(w, batcheddata, means, stddevs)
    yhats, ygolds = predictiongenerator(w, batcheddata, means, stddevs)
    lossamount = mean(abs2, ygolds - yhats)
    lossRWamount = mean(abs2, ygolds .- 1)
    R2 = coefficientofdetermination(ygolds, yhats)
    yearlyreturn, acc = returncalc(ygolds, yhats)
    monthlyleveragereturn = returncalc(ygolds, yhats, leverage = 10)[1] ^(1/12)
    DMstat = dieboldmariano(ygolds, yhats)
    significance = cdf(Normal(), DMstat)
    println("Loss: ", lossamount, " RWloss: ", lossRWamount)
    println("R2: ", R2, " Accuracy: ", acc)
    println("prodreturn: ", yearlyreturn, " monthlyleverage: ", monthlyleveragereturn)
    println("DM Stat: ", DMstat, " Significance: ", significance)
end

seed = 5
hiddensize = 90
layernum = 1
rnntype = :gru
batchlength = 100
trainlength = 3000
devlength = 100
testlength = 100
epochs = 100

gaussianinput = 0.1
dropoutinput = 0.1
dropouthidden = 0.3
srand(seed)
inputs, ygold, means, stddevs = initdatadaily(trnlength = trainlength, devtstlength = devlength + testlength)
trn, dev, tst = minibatcher(inputs, ygold, batchlength, trainlength, devlength, testlength)
w = weightsinit(inputs, hiddensize, layernum, rnntype, dropouthidden)

for i = 1:epochs
    h = hiddeninit(w)
    train(w, trn, h, gaussianinput, dropoutinput, dropouthidden)
    println("Epoch Number: ", i)
    println("Training Set Stats:")
    report(w, trn, means, stddevs)
    println("Development Set Stats:")
    report(w, dev, means, stddevs)
    println("Test Set Stats:")
    report(w, tst, means, stddevs)
    println("")
end
