using Knet

function initdatadaily(;trnlength = 2000, devandtstlength = 225)
    i = 4
    rawdata = Array{Float32,2}(readdlm("rawdata.txt"))
    #rawdata = rawdata[[4,9,10],:]
    data, meanvec, stddev = datapreprocess(rawdata, i, trnlength)
    if trnlength + devandtstlength + 1 <= size(data, 2)
        x = data[:, 1:trnlength + devandtstlength]
        y = data[i, 2:trnlength + devandtstlength + 1]
    end
    if trnlength + devandtstlength + 1 > size(data, 2)
        println("Not enough observations are present in the dataset.")
        error("Not enough observations are present in the dataset.")
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
    #total = vcat(logdata, #=changedata, =#rawdata[:,2:end])
    data, meansample, stdsample = standardization(total, i, trnlength)
    return data, meansample, stdsample
end

function standardization(total, target, trnlength)
    statdata = total[:, 1:trnlength]
    meansample = mean(statdata, 2)
    stdsample = std(statdata, 2, corrected = false)
    return (total .- meansample) ./ stdsample, meansample[target], stdsample[target]
end

function minibatch(inputs, ygold, batchlength, trainlength, devlength, testlength)
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

function weightsinit(inputs, hiddensize, layernum, rnntype)
    winit = 0.1
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

function predict(w, input, h; gaussianinput = 0, gaussianhidden = 0, dropoutinput = 0, dropouthidden = 0)
    hx, cx = h
    r, wr, wy, by = w
    inputnoisevector = gpu() == -1 ? Array{Float32}(gaussian(size(input), mean = 1, std = gaussianinput)) : KnetArray{Float32}(gaussian(size(input), mean = 1, std = gaussianinput))
    input = input .* inputnoisevector
    input = dropout(input, dropoutinput)
    hiddens, hx, cx, = rnnforw(r, wr, input, hx, cx)
    h[1] = hx
    h[2] = cx
    hiddens = reshape(hiddens, size(hiddens,1), :)
    hiddennoisevector = gpu() == -1 ? Array{Float32}(gaussian(size(hiddens), mean = 1, std = gaussianhidden)) : KnetArray{Float32}(gaussian(size(hiddens), mean = 1, std = gaussianhidden))
    hiddens = hiddens .* hiddennoisevector
    hiddens = dropout(hiddens, dropouthidden)
    yhat = wy * hiddens .+ by
    return yhat[:]
end

function loss(w, data, h, gaussianx, gaussianh, dropoutx, dropouth)
    input, ygold = data
    yhat = predict(w, input, h, gaussianinput = gaussianx, gaussianhidden = gaussianh, dropoutinput = dropoutx, dropouthidden = dropouth)
    1/2 * mean(abs2, yhat - ygold)
end

lossgrad = grad(loss)

function train(w, batches, h, gaussianinput, gaussianhidden, dropoutinput, dropouthidden)
    optim = optimizers(w, Adadelta)
    for data in batches
        gradamount = lossgrad(w, data, h, gaussianinput, gaussianhidden, dropoutinput, dropouthidden)
        update!(w, gradamount, optim)
    end
end

function reversestandardization(input, means, stddevs)
    input * stddevs .+ means
end

function returnensemble(yhat, ygold, datatype; leverage = 1, commissions = 0 / 10000)
    if datatype == :ln
        yhat = exp.(yhat)
        ygold = exp.(ygold)
    end
    yact = copy(yhat)
    yact[yact .< 1] = -1
    yact[yact .> 1] = 1
    marketchange = (ygold #=./ yt=# .- 1) * leverage
    realized = yact .* marketchange
    accseries = copy(realized)
    accseries[accseries .< 0] = 0
    accseries[accseries .> 0] = 1
    prodreturnyearly = prod(realized .+ 1)^(253/length(realized)) * (1 - 2 * commissions * leverage)^253
    return prodreturnyearly, mean(accseries)
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

function ensemblesignificance(yhat, ygold, datatype)
    if datatype == :ln
        yhat = exp.(yhat)
        ygold = exp.(ygold)
    end
    modelerror = ygold .- yhat
    randomerror = ygold .- 1
    modelloss = modelerror .^ 2 * 1/2
    randomloss = randomerror .^2 * 1/2
    lossdiffs = randomloss - modelloss
    dbar = mean(lossdiffs)
    t = length(lossdiffs)
    gamma = mean((lossdiffs .- dbar).^2)
    stat = dbar / ((gamma/t)^(1/2))
    return stat
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

function teststats(w, tst, means, stddevs)
    if isempty(tst) == 1
        return nothing
    end
#    tstloss, tstRWloss = stats(w, tst, means, stddevs)
#    tstR2 = coefficientofdetermination(w, tst, means, stddevs)
    tstprodreturn, tstacc, tstsumreturn = returncalc(w, tst, means, stddevs)
    tstmonthlyleverage = returncalc(w, tst, means, stddevs, leverage = 10)[1]^(1/12)
    tstDMstat = significance(w, tst, means, stddevs)
    #=println("tstloss: ", tstloss)
    println("tst R2: ", tstR2)
    println("tstacc: ", tstacc)
    println("tstRWloss: ", tstRWloss)
    println("tstprodreturn: ", tstprodreturn)
    println("tstmonthlyleverage: ", tstmonthlyleverage)#, :trnsumreturn , trnsumreturn)
    println("tstDMstat: ", tstDMstat)=#
    return tstDMstat, tstacc, tstmonthlyleverage
end

function report(w, trn, dev, epoch, means, stddevs)
#    trnloss, trnRWloss = stats(w, trn, means, stddevs)
#    devloss, devRWloss = stats(w, dev, means, stddevs)
#    trnR2 = coefficientofdetermination(w, trn, means, stddevs)
#    devR2 = coefficientofdetermination(w, dev, means, stddevs)
    trnprodreturn, trnacc, trnsumreturn = returncalc(w, trn, means, stddevs)
    devprodreturn, devacc, devsumreturn = returncalc(w, dev, means, stddevs)
    trnmonthlyleverage = returncalc(w, trn, means, stddevs, leverage = 10)[1]^(1/12)
    devmonthlyleverage = returncalc(w, dev, means, stddevs, leverage = 10)[1]^(1/12)
    trnDMstat = significance(w, trn, means, stddevs)
    devDMstat = significance(w, dev, means, stddevs)
    #=println("epoch: ", epoch)
    println("trnloss: ", trnloss, " devloss: ", devloss)
    println("trn R2: ", trnR2, " dev R2: ", devR2)
    println("trnacc: ", trnacc, " devacc: ", devacc)
    println("trnRWloss: ", trnRWloss , " devRWloss: ", devRWloss)
    println("trnprodreturn: ", trnprodreturn, " devprodreturn: ", devprodreturn)
    println("trnmonthlyleverage: ", trnmonthlyleverage, " devmonthlyleverage: ", devmonthlyleverage)#, :trnsumreturn , trnsumreturn)
    println("trnDMstat: ", trnDMstat, " devDMstat: ", devDMstat)=#
    return trnDMstat, devDMstat, trnacc, devacc, trnmonthlyleverage, devmonthlyleverage
end

function main(seed, hiddensize, layernum, rnntype, trainlength, devlength, testlength)
    srand(seed)
    batchlength = 90
    gaussianinput = 0#.33
    gaussianhidden = 0#.66
    dropoutinput = 0#.1
    dropouthidden = 0#.2
    epocs = 50
    totalepochspassed = 0
    inputs, ygold, means, stddevs = initdatadaily(trnlength = trainlength, devandtstlength = devlength + testlength)
    w = weightsinit(inputs, hiddensize, layernum, rnntype)
    trn, dev, tst = minibatch(inputs, ygold, batchlength, trainlength, devlength, testlength)
    report(w, trn, dev, 0, means, stddevs)
    teststats(w, tst, means, stddevs)
    modelmax = recordingweights(w)
    trnDMmax = 0
    devDMmax = 0
    tstDMmax = 0
    trnaccmax = 0
    devaccmax = 0
    tstaccmax = 0
    trnmonthlyleveragemax = 0
    devmonthlyleveragemax = 0
    tstmonthlyleveragemax = 0
    epochDMmax = 0
    for i = 1:epocs
        h = hiddeninit(w)
        train(w, trn, h, gaussianinput, gaussianhidden, dropoutinput, dropouthidden)
        trnDM, devDM, trnacc, devacc, trnmonthlyleverage, devmonthlyleverage = report(w, trn, dev, i, means, stddevs)
        tstDM, tstacc, tstmonthlyleverage  = teststats(w, tst, means, stddevs)
        if devDM > devDMmax
            modelmax = recordingweights(w)
            trnDMmax = trnDM
            devDMmax = devDM
            tstDMmax = tstDM
            trnaccmax = trnacc
            devaccmax = devacc
            tstaccmax = tstacc
            trnmonthlyleveragemax = trnmonthlyleverage
            devmonthlyleveragemax = devmonthlyleverage
            tstmonthlyleveragemax = tstmonthlyleverage
            epochDMmax = i
        end
        totalepochspassed += 1
    end
    while epochDMmax > totalepochspassed - 50
        for i = 1 + totalepochspassed : epocs + totalepochspassed
            h = hiddeninit(w)
            train(w, trn, h, gaussianinput, gaussianhidden, dropoutinput, dropouthidden)
            trnDM, devDM, trnacc, devacc, trnmonthlyleverage, devmonthlyleverage = report(w, trn, dev, i, means, stddevs)
            tstDM, tstacc, tstmonthlyleverage = teststats(w, tst, means, stddevs)
            if devDM > devDMmax
                modelmax = recordingweights(w)
                trnDMmax = trnDM
                devDMmax = devDM
                tstDMmax = tstDM
                trnaccmax = trnacc
                devaccmax = devacc
                tstaccmax = tstacc
                trnmonthlyleveragemax = trnmonthlyleverage
                devmonthlyleveragemax = devmonthlyleverage
                tstmonthlyleveragemax = tstmonthlyleverage
                epochDMmax = i
            end
            totalepochspassed += 1
        end
    end
    return modelmax, trnDMmax, devDMmax, tstDMmax, trnaccmax, devaccmax, tstaccmax, trnmonthlyleveragemax, devmonthlyleveragemax, tstmonthlyleveragemax, epochDMmax
end

function ensemblesearch(noofseeds, structures, rnntype, condition, trainlength, devlength, testlength)
    modelstobeensembled = Any[]
    for structure in structures
        hiddensize, layernum = structure
        println("A ", layernum, " layered ", rnntype, " network with ", hiddensize, " nodes in each layer:")
            for seed = 1:noofseeds
                model, trnDMmax, devDMmax, tstDMmax, trnaccmax, devaccmax, tstaccmax, trnmonthlyleveragemax, devmonthlyleveragemax, tstmonthlyleveragemax, epochDMmax = main(seed, hiddensize, layernum, rnntype, trainlength, devlength, testlength)
                if condition == :none && trnDMmax > 0 && devDMmax > 0
                    push!(modelstobeensembled, model)
                end
                if condition == :dev &&  trnDMmax > 1.65 && devDMmax > 1.65
                    push!(modelstobeensembled, model)
                end
                if condition == :both && trnDMmax > 1.96 && devDMmax > 1.96
                    push!(modelstobeensembled, model)
                end
                println("seed: ", seed, " trnDMmax: ", trnDMmax, " devDMmax: ", devDMmax, " tstDMmax: ", tstDMmax, " epochDMmax: ", epochDMmax)
                println("seed: ", seed, " trnaccmax: ", trnaccmax, " devaccmax: ", devaccmax, " tstaccmax: ", tstaccmax, " epochDMmax: ", epochDMmax)
                println("seed: ", seed, " trnmonthlyleveragemax: ", trnmonthlyleveragemax, " devmonthlyleveragemax: ", devmonthlyleveragemax, " tstmonthlyleveragemax: ", tstmonthlyleveragemax, " epochDMmax: ", epochDMmax)
                println(" ")
            end
    end
    return modelstobeensembled
end

function finalensembler(seeds, structures, rnntype, condition, trainlength, devlength, testlength)
    modelstobeensembled = ensemblesearch(seeds, structures, rnntype, condition, trainlength, devlength, testlength)
    return modelstobeensembled
end

function recordingweights(w)
    rb, wrb, wyb, byb = w
    r = rb
    wr = copy(wrb)
    wy = copy(wyb)
    by = copy(byb)
    return r, wr, wy, by
end

function emsembledpredictions(modelstobeensembled, batcheddata, means, stddevs)
    ygoldsTotal = Any[]
    yhatsTotal = Any[]
    ygoldsExpTotal = Any[]
    yhatsExpTotal = Any[]
    for w in modelstobeensembled
        h = hiddeninit(w)
        ygolds = Any[]
        yhats = Any[]
        for data in batcheddata
            input, ygold = data
            yhat = predict(w, input, h)
            ygold = Array(reversestandardization(ygold, means, stddevs))
            yhat = Array(reversestandardization(yhat, means, stddevs))
            push!(ygolds,ygold)
            push!(yhats,yhat)
        end
        ygolds = vcat(ygolds...)
        yhats = vcat(yhats...)
        push!(ygoldsTotal, ygolds)
        push!(yhatsTotal, yhats)
        push!(ygoldsExpTotal, exp.(ygolds))
        push!(yhatsExpTotal, exp.(yhats))
    end
    ygoldsEnsembled = mean(hcat(ygoldsTotal...), 2)
    yhatsEnsembled = mean(hcat(yhatsTotal...), 2)
    ygoldsExpEnsembled = mean(hcat(ygoldsExpTotal...), 2)
    yhatsExpEnsembled = mean(hcat(yhatsExpTotal...), 2)
    return #=ygoldsEnsembled, yhatsEnsembled,=# ygoldsExpEnsembled, yhatsExpEnsembled
end


function mainmain()
    for i = 1700:100:1900
        trainlength = i
        devlength = 225
        testlength = 100
        batchlength = 90
        seednumber = 5
        structures = [(30,3),(90,1),(50,2)]
        rnntype = :gru
        condition = :dev
        println("For the training that takes first ", i, " observations as trn data:")
        println(" ")
        modelstobeensembled = finalensembler(seednumber, structures, rnntype, condition, trainlength, devlength, testlength)
        inputs, ygold, means, stddevs = initdatadaily(trnlength = trainlength, devandtstlength = devlength + testlength)
        trn, dev, tst = minibatch(inputs, ygold, batchlength, trainlength, devlength, testlength)
        if length(modelstobeensembled) > 0
            ensembledTRNygolds, ensembledTRNyhats = emsembledpredictions(modelstobeensembled, trn, means, stddevs)
            ensembledDEVygolds, ensembledDEVyhats = emsembledpredictions(modelstobeensembled, dev, means, stddevs)
            ensembledTSTygolds, ensembledTSTyhats = emsembledpredictions(modelstobeensembled, tst, means, stddevs)

            yearlyTRNreturn, trnAcc =  returnensemble(ensembledTRNyhats, ensembledTRNygolds, :exp, leverage = 1)
            monthlyTRNleverage = returnensemble(ensembledTRNyhats, ensembledTRNygolds, :exp, leverage = 10)[1]^(1/12)
            DMstatTRN = ensemblesignificance(ensembledTRNyhats, ensembledTRNygolds, :exp)

            yearlyDEVreturn, devAcc =  returnensemble(ensembledDEVyhats, ensembledDEVygolds, :exp, leverage = 1)
            monthlyDEVleverage = returnensemble(ensembledDEVyhats, ensembledDEVygolds, :exp, leverage = 10)[1]^(1/12)
            DMstatDEV = ensemblesignificance(ensembledDEVyhats, ensembledDEVygolds, :exp)

            yearlyTSTreturn, tstAcc =  returnensemble(ensembledTSTyhats, ensembledTSTygolds, :exp, leverage = 1)
            monthlyTSTleverage = returnensemble(ensembledTSTyhats, ensembledTSTygolds, :exp, leverage = 10)[1]^(1/12)
            DMstatTST = ensemblesignificance(ensembledTSTyhats, ensembledTSTygolds, :exp)

            println("")
            println("Ensemble Stats:")
            println("DMstatTRN: ", DMstatTRN, " DMstatDEV: ", DMstatDEV, " DMstatTST: ", DMstatTST)
            println("trnAcc: ", trnAcc, " devAcc: ", devAcc, " tstAcc: ", tstAcc)
            println("yearlyTRNreturn: ", yearlyTRNreturn, " yearlyDEVreturn: ", yearlyDEVreturn, " yearlyTSTreturn: ", yearlyTSTreturn)
            println("monthlyTRNleverage: ", monthlyTRNleverage, " monthlyDEVleverage: ", monthlyDEVleverage, " monthlyTSTleverage: ", monthlyTSTleverage)
            println("")
        end
        if length(modelstobeensembled) == 0
            println("No predictions for this case.")
            println("")
        end
    end
end

mainmain()
#info("Batches are created.")
#trnDM, devDM, trnacc, devacc, trnmonthlyleverage, devmonthlyleverage = report(modelstobeensembled[1], trn, dev, 0, means, stddevs)
#tstDM, tstacc, tstmonthlyleverage = teststats(modelstobeensembled[1], tst, means, stddevs)
#=function search()
    layers = [1, 2, 3, 4, 5]
    hiddensizes = [10,30,50,90,170]
    rnntypes = [:gru, :lstm]
    for layer in layers
        for hiddensize in hiddensizes
            for rnntype in rnntypes
                println("A ", layer, " layered ", rnntype, " network with ", hiddensize, " nodes in each layer:")
                ensemblesearch(15, hiddensize, layer, rnntype)
                println("")
            end
        end
    end
end

search()=#
