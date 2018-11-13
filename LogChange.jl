using Knet

function initdatadaily(;trnlength = 2000, tstlength = 225)
    i = 4
    rawdata = Array{Float32,2}(readdlm("rawdata.txt"))
    #rawdata = rawdata[[4,9,10],:]
    data, meanvec, stddev = datapreprocess(rawdata, i, trnlength)
    if trnlength + tstlength + 1 <= size(data, 2)
        x = data[:, 1:trnlength + tstlength]
        y = data[i, 2:trnlength + tstlength + 1]
    end
    if trnlength + tstlength + 1 >= size(data, 2)
        x = data[:,1:end-1]
        y = data[i,2:end]
    end
    return x, y, meanvec, stddev
end

function changecalc(data)
    if ndims(data) == 1
        len = length(data)
        yt1 = data[2:end]
        yt0 = data[1:end-1]
        return yt1 ./ yt0
    end
    if ndims(data) == 2
        len = size(data,2)
        yt1 = data[:,2:end]
        yt0 = data[:,1:end-1]
        return yt1 ./ yt0
    end
end

function datapreprocess(rawdata, i, trnlength)
    changedata = changecalc(rawdata)
    logdata = log.(changedata)
    total = vcat(logdata, changedata, rawdata[:,2:end])
    data, meansample, stdsample = standardization(total, i, trnlength)
    return data, meansample, stdsample
end

function standardization(rawdata, target, trnlength)
    statdata = rawdata[:, 1:trnlength]
    meansample = mean(statdata, 2)
    stdsample = std(statdata, 2, corrected = false)
    return (rawdata .- meansample) ./ stdsample, meansample[target], stdsample[target]
end

function minibatch(inputs, ygold, batchlength, testlength)
    len = size(inputs, 2)
    inputdim = size(inputs, 1)
    trn = []
    tst = []
    for i = 1 :batchlength:len - testlength
        bl = min(i + batchlength -1, len - testlength)
        input = inputs[:,[i:bl...]]
        input = reshape(input, inputdim, 1, :)
        ygol = ygold[i:bl]
        if gpu() != -1
            input = convert(KnetArray,input)
            ygol = convert(KnetArray, ygol)
        end
        push!(trn, (input,ygol))
    end
    for i = len - testlength + 1 :batchlength: len
        bl = min(len, i + batchlength -1)
        input = inputs[:,[i:bl...]]
        input = reshape(input, inputdim, 1, :)
        ygol = ygold[i:bl]
        if gpu() != -1
            input = convert(KnetArray, input)
            ygol = convert(KnetArray, ygol)
        end
        push!(tst, (input,ygol))
    end
    return trn, tst
end

function weightsinit(inputs)
    hiddensize = 50
    winit = 0.1
    layernum = 1
    rnntype = :lstm
    inputsize =  size(inputs, 1)
    r, wr = rnninit(inputsize, hiddensize, rnnType = rnntype, numLayers = layernum)
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

function predict(w, input, h, dropoutinput, dropouthidden)
    hx, cx = h
    r, wr, wy, by = w
    input = dropout(input, dropoutinput)
    hiddens, hx, cx, = rnnforw(r, wr, input, hx, cx)
    h[1] = hx
    h[2] = cx
    hiddens = reshape(hiddens, size(hiddens,1), :)
    hiddens = dropout(hiddens, dropouthidden)
    yhat = wy * hiddens .+ by
    return yhat[:]
end

function loss(w, data, h, dropoutinput, dropouthidden)
    input, ygold = data
    yhat = predict(w, input, h, dropoutinput, dropouthidden)
    1/2 * mean(abs2, yhat - ygold)
end

lossgrad = grad(loss)

function train(w, batches, h, dropoutinput, dropouthidden)
    optim = optimizers(w, Adadelta)
    for data in batches
        gradamount = lossgrad(w, data, h, dropoutinput, dropouthidden)
        update!(w, gradamount, optim)
    end
end

function reversestandardization(input, means, stddevs)
    input * stddevs .+ means
end

function returncalc(w, batcheddata, means, stddevs; leverage = 1, commissions = 0 / 10000)
    returns = Any[]
    h = hiddeninit(w)
    for data in batcheddata
        input, ygold = data
        #yt = reshape(input, size(input,1),:)[4,:]
        #yt = input[4,:,:][:]
        yhat = predict(w, input, h)
        ygold = exp.(reversestandardization(ygold, means, stddevs))
        #yt = reversestandardization(yt, means, stddevs)
        yhat = exp.(reversestandardization(yhat, means, stddevs))
        #yact = yhat - yt
        yact = copy(yhat)
        yact[yact .< 1] = -1
        yact[yact .> 1] = 1
        marketchange = (ygold #=./ yt=# .- 1) * leverage
        realized = yact .* marketchange
        push!(returns, realized)
    end
    returnseries = vcat(returns...)
    accseries = copy(returnseries)
    accseries[accseries .< 0] = 0
    accseries[accseries .> 0] = 1
    prodreturnyearly = prod(returnseries .+ 1)^(253/length(returnseries)) * (1 - 2 * commissions * leverage)^253
    sumreturnyearly = mean(returnseries .+ 1)^253
    return prodreturnyearly, mean(accseries), sumreturnyearly
end

function significance(w, batcheddata, means, stddevs)
    h = hiddeninit(w)
    lossdifferentialseries = Any[]
    for data in batcheddata
        input, ygold = data
        #yt = reshape(input, size(input,1),:)[4,:]
        #yt = input[4,:,:][:]
        yhat = predict(w, input, h)
        ygold = exp.(reversestandardization(ygold, means, stddevs))
        #yt = reversestandardization(yt, means, stddevs)
        yhat = exp.(reversestandardization(yhat, means, stddevs))
        yhat = Array(yhat)
        ygold = Array(ygold)
        #yt = Array(yt)
        modelerror = ygold .- yhat
        randomerror = ygold .- 1
        modelloss = modelerror .^2 * 1/2
        randomloss = randomerror .^2 * 1/2
        lossdiffs = randomloss - modelloss
        push!(lossdifferentialseries, lossdiffs)
    end
    losses = vcat(lossdifferentialseries...)
    dbar = mean(losses)
    t = length(losses)
    gamma = mean((losses .- dbar).^2)
    stat = dbar / ((gamma/t)^(1/2))
    return stat
end

function coefficientofdetermination(w, batcheddata, means, stddevs)
    ygolds = Any[]
    yhats = Any[]
    h = hiddeninit(w)
    for data in batcheddata
        input, ygold = data
        #yt = reshape(input, size(input,1),:)[4,:]
        #yt = input[4,:,:][:]
        yhat = predict(w, input, h)
        ygold = exp.(reversestandardization(ygold, means, stddevs))
        #yt = reversestandardization(yt, means, stddevs)
        yhat = exp.(reversestandardization(yhat, means, stddevs))
        push!(ygolds, ygold)
        push!(yhats, yhat)
    end
    ygolds = Array(vcat(ygolds...))
    yhats = Array(vcat(yhats...))
    ymean = mean(ygolds)
    SSTotal = sum((ygolds .- ymean).^2)
    SSRes = sum((ygolds .- yhats).^2)
    return 1 - SSRes/SSTotal
end

function stats(w, batcheddata, means, stddevs)
    totalloss = 0.0
    totalRWloss = 0.0
    totalcount = 0
    h = hiddeninit(w)
    for data in batcheddata
        input, ygold = data
        #yt = reshape(input, size(input,1),:)[4,:]
        #yt = input[4,:,:][:]
        yhat = predict(w, input, h)
        ygold = exp.(reversestandardization(ygold, means, stddevs))
        #yt = reversestandardization(yt, means, stddevs)
        yhat = exp.(reversestandardization(yhat, means, stddevs))
        lossamount = 1/2 * mean(abs2, ygold - yhat)
        lossRWamount = 1/2 * mean(abs2, ygold .- 1)
        batchlength = length(data[2])
        totalloss += lossamount * batchlength
        totalRWloss += lossRWamount * batchlength
        totalcount += batchlength
    end
    totalloss / totalcount, totalRWloss / totalcount
end

function report(w, trn, tst, epoch, means, stddevs)
    trnloss, trnRWloss = stats(w, trn, means, stddevs)
    tstloss, tstRWloss = stats(w, tst, means, stddevs)
    trnR2 = coefficientofdetermination(w, trn, means, stddevs)
    tstR2 = coefficientofdetermination(w, tst, means, stddevs)
    trnprodreturn, trnacc, trnsumreturn = returncalc(w, trn, means, stddevs)
    tstprodreturn, tstacc, tstsumreturn = returncalc(w, tst, means, stddevs)
    trnmonthlyleverage = returncalc(w, trn, means, stddevs, leverage = 10)[1]^(1/12)
    tstmonthlyleverage = returncalc(w, tst, means, stddevs, leverage = 10)[1]^(1/12)
    trnDMstat = significance(w, trn, means, stddevs)
    tstDMstat = significance(w, tst, means, stddevs)
    println("epoch: ", epoch)
    println("trnloss: ", trnloss, " tstloss: ", tstloss)
    println("trn R2: ", trnR2, " tst R2: ", tstR2)
    println("trnacc: ", trnacc, " tstacc: ", tstacc)
    println("trnRWloss: ", trnRWloss , " tstRWloss: ", tstRWloss)
    println("trnprodreturn: ", trnprodreturn, " tstprodreturn: ", tstprodreturn)
    println("trnmonthlyleverage: ", trnmonthlyleverage, " tstmonthlyleverage: ", tstmonthlyleverage)#, :trnsumreturn , trnsumreturn)
    println("trnDMstat: ", trnDMstat, " tstDMstat: ", tstDMstat)
    #println((trnloss * 2)^(1/2))
end

function main()
    srand(2)
    batchlength = 90
    testlength = 225
    dropoutinput = 0.2
    dropouthidden = 0.5
    epocs = 800
    inputs, ygold, means, stddevs = initdatadaily()
    w = weightsinit(inputs)
    trn, tst = minibatch(inputs, ygold, batchlength, testlength)
    info("Batches are created.")
    report(w, trn, tst, 0, means, stddevs)
    info("Report 0.")
    for i = 1:epocs
        h = hiddeninit(w)
        train(w, trn, h, dropoutinput, dropouthidden)
        report(w, trn, tst, i, means, stddevs)
    end
end

main()
