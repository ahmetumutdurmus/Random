push!(LOAD_PATH, pwd())
using ExcelReaders, Dates, DateHandler, PrincipalComponentsAnalysis, JuMP, Cbc, StatsBase, LinearAlgebra, Statistics, DataStructures, XLSX, ResultsReporter, PyCall
@pyimport openpyxl
"""
    CustomDataInit(;first::Symbol = Symbol("2011Q1"), last::Symbol = Symbol(string(year(today())) * "Q" * string(quarterofyear(today())))
My custom data reader. Options `first` and `last` determine the data range. The default is the whole available range up to date.
"""
function CustomDataInit(o::Dict; first::Symbol = Symbol("2011Q1"), last::Symbol = Symbol(string(year(today())) * "Q" * string(quarterofyear(today()))))
    if o[:dataset] == :Bigset
        PriceSheet = readxlsheet("Close & PE.xlsx", "BigSetPrice")
        PESheet = readxlsheet("Close & PE.xlsx", "BigSetPE")
        ntradeables = 107
    elseif o[:dataset] == :Smallset
        PriceSheet = readxlsheet("Close & PE.xlsx", "SmallSetPrice")
        PESheet = readxlsheet("Close & PE.xlsx", "SmallSetPE")
        ntradeables = 68
    end
    LevelData = PriceSheet[2:end, 2:end]
    ReturnData = LevelData[1:end-1,:] ./ LevelData[2:end,:] .- 1
    LevelData = LevelData[1:end-1,:]
    DateVec = Date.(PriceSheet[2:end-1,1])
    StockNames = PriceSheet[1, 2:end]
    PERatios = PESheet[2:end-1, 2:end]
    QuarterInfo = Symbol.(string.(year.(DateVec)) .* "Q" .* string.(quarterofyear.(DateVec)))
    indexes = (QuarterInfo .>= first) .& (QuarterInfo .<= last)
    data = Dict()
    data[:dates] =  DateVec[indexes, :][:]
    data[:PEs] = PERatios[indexes, :].^-1
    data[:stocks] = StockNames[1:ntradeables]
    data[:stocklevels] = Array{Float64}(LevelData[indexes, 1:ntradeables])
    data[:stockreturns] = Array{Float64}(ReturnData[indexes, 1:ntradeables])
    data[:securities] = StockNames[ntradeables+1:end]
    data[:securitieslevels] = Array{Float64}(LevelData[indexes, ntradeables+1:end])
    data[:securitiesreturns] = Array{Float64}(ReturnData[indexes, ntradeables+1:end])
    data
end

"""
    rankstatextractor(data::Dict, o::Dict)
"""
function rankstatextractor(data::Dict, o::Dict)
    _, RetroDataIndex = DateHandler.DateRetrospection(data[:dates], o[:endofperiod], o[:retrospectionperiod])
    RelevantEP = data[:PEs][RetroDataIndex, :][1,:]
    PEranks = sortperm(sortperm(RelevantEP, rev = true))
    RelevantReturn = o[:addsecurities] ? hcat(data[:stockreturns], data[:securitiesreturns])[RetroDataIndex, :] : data[:stockreturns][RetroDataIndex, :]
    reconstructed = inversePCA(PCA(RelevantReturn, typeofSim = o[:pcatype], normtype = o[:normtype], featurenum = o[:featurenum]))[1:o[:PCAdays], 1:size(data[:stockreturns], 2)]'
    realized = RelevantReturn[1:o[:PCAdays], 1:size(data[:stockreturns], 2)]'
    reconstructed = o[:pcanorm] ? reconstructed ./ colwisenorm(reconstructed) .* colwisenorm(realized) : reconstructed
    returnspread = reconstructed - realized
    returnspreadndays = prod(returnspread .+ 1, dims = 2) .- 1
    spreadrankings = sortperm(sortperm(returnspreadndays[1:end], rev = true))
    collectiverankstat = spreadrankings + o[:lambda] * PEranks
    if o[:rankmode] == :Collective
        return collectiverankstat
    end
    if o[:rankmode] == :PEranks
        return PEranks
    end
    if o[:rankmode] == :Spreadranks
        return spreadrankings
    end
end

"""
"""
function RankingNExplicitChurn(data::Dict, o::Dict)
    rankstat = rankstatextractor(data, o)
    m = Model(solver = CbcSolver())
    @variable(m, 0 <= x[1:length(rankstat)] <= 1/o[:numberofstocks])
    @objective(m, Min, dot(x, rankstat))
    @constraint(m, sum(x) == 1)
    solve(m)
    output = reshape(Array(getvalue(x)), 1, :)
    return output
end

"""
"""
function RankingNExplicitChurn(data::Dict, o::Dict, oldportfolio)
    rankstat = rankstatextractor(data, o)
    m = Model(solver = CbcSolver())
    @variable(m, 0 <= x[1:length(rankstat)] <= 1/o[:numberofstocks])
    @variable(m, z[1:length(rankstat)])
    @objective(m, Min, dot(x, rankstat))
    @constraint(m, sum(x) == 1)
    @constraint(m, [i = 1:length(x)], z[i] - x[i] >= -oldportfolio[i])
    @constraint(m, [i = 1:length(x)], z[i] + x[i] >= oldportfolio[i])
    @constraint(m, sum(z) <= 2*o[:churn])
    solve(m)
    output = reshape(Array(getvalue(x)), 1, :)
    return output
end

"""
    portfolioconstructor(data::Dict, o::Dict)
Create a portfolio allocation matrix given data and options.
"""
function portfolioconstructor(data::Dict, o::Dict)
    PortfolioAllocations = Any[]
    o[:endofperiod] = o[:initialstart]
    oldportfolio = RankingNExplicitChurn(data, o)
    push!(PortfolioAllocations, oldportfolio)
    datevec = data[:dates]
    for date in datevec[datevec .>= o[:initialstart]][end-1:-1:1]
        o[:endofperiod] = date
        newportfolio = RankingNExplicitChurn(data, o, oldportfolio)
        push!(PortfolioAllocations, newportfolio)
        oldportfolio = newportfolio
    end
    PortfolioAllocations = reverse(vcat(PortfolioAllocations...), dims = 1)
    return PortfolioAllocations
end

colwisenorm(A) = sqrt.(sum(abs2, A, dims = 1))

function readlastportfolio(data; dt = data[:dates][1])
    ws = first(data[:dates][data[:dates] .< dt])
    y, m, d = yearmonthday(ws)
    wsn = string.(d, ".",m, ".",y)
    stocknames = readxl("CRE Allocations.xlsx", wsn * "!A3:A22")[:]
    oldportfolio = zeros(size(data[:stocks]))
    oldportfolio[indexin(stocknames, data[:stocks])] .= 1/20
    oldportfolio, stocknames
end

function dailycreroutine()
    o = Dict()
    o[:dataset] = :Bigset
    o[:retrospectionperiod] = Month(6)
    o[:numberofstocks] = 20
    o[:PCAdays] = 4
    o[:rankmode] = :Collective # {:Collective, :PEranks, :Spreadranks}
    o[:addsecurities] = false
    o[:pcatype] = :cov
    o[:pcanorm] = false
    o[:normtype] = 2 #{:nonorm, :meannorm, :zscore, :minmax} Note that if minmax is used using range[-1,1] or [0.1,0.9] is also an option need 2 take care of that.
    o[:featurenum] = 8 #{1:NumberOfTradeables}
    o[:lambda] = 0.40 #[0, +)
    o[:churn] = 0.15 #[0.05, 0.3]
    o[:initialstart] = Date(2013,1,1)
    data = CustomDataInit(o)
    oldportfolio, currentstocks = readlastportfolio(data)
    o[:endofperiod] = today() + Day(1)
    _, RetroDataIndex = DateHandler.DateRetrospection(data[:dates], o[:endofperiod], o[:retrospectionperiod])
    RelevantEP = data[:PEs][RetroDataIndex, :][1,:]
    PEranks = sortperm(sortperm(RelevantEP, rev = true))
    RelevantReturn = o[:addsecurities] ? hcat(data[:stockreturns], data[:securitiesreturns])[RetroDataIndex, :] : data[:stockreturns][RetroDataIndex, :]
    reconstructed = inversePCA(PCA(RelevantReturn, typeofSim = o[:pcatype], normtype = o[:normtype], featurenum = o[:featurenum]))[1:o[:PCAdays], 1:size(data[:stockreturns], 2)]'
    realized = RelevantReturn[1:o[:PCAdays], 1:size(data[:stockreturns], 2)]'
    reconstructed = o[:pcanorm] ? reconstructed ./ colwisenorm(reconstructed) .* colwisenorm(realized) : reconstructed
    returnspread = reconstructed - realized
    returnspreadndays = prod(returnspread .+ 1, dims = 2) .- 1
    spreadrankings = sortperm(sortperm(returnspreadndays[1:end], rev = true))
    rankstat = spreadrankings + o[:lambda] * PEranks
    cumrank = sortperm(sortperm(rankstat, rev = false))
    m = Model(solver = CbcSolver())
    @variable(m, 0 <= x[1:length(rankstat)] <= 1/o[:numberofstocks])
    @variable(m, z[1:length(rankstat)])
    @objective(m, Min, dot(x, rankstat))
    @constraint(m, sum(x) == 1)
    @constraint(m, [i = 1:length(x)], z[i] - x[i] >= -oldportfolio[i])
    @constraint(m, [i = 1:length(x)], z[i] + x[i] >= oldportfolio[i])
    @constraint(m, sum(z) <= 2*o[:churn])
    solve(m)
    output = reshape(Array(getvalue(x)), 1, :)[:]
    longs = collect(1:length(output))[output .> 0]
    shorts = collect(1:length(output))[output .<= 0]
    imatlong = hcat(data[:stocks], reconstructed, realized, returnspread, returnspreadndays, spreadrankings, RelevantEP, PEranks, rankstat, cumrank)[longs,:]
    imatshort = hcat(data[:stocks], reconstructed, realized, returnspread, returnspreadndays, spreadrankings, RelevantEP, PEranks, rankstat, cumrank)[shorts,:]
    imat = vcat(imatlong[sortperm(rankstat[longs], rev = false), :], imatshort[sortperm(rankstat[shorts], rev = false), :])
    wb = openpyxl.load_workbook("CRE Allocations.xlsx")
    source = wb[:active]
    y, m, d = yearmonthday(o[:endofperiod] - Day(1))
    wsn = string.(d, ".",m, ".",y)
    wb[:copy_worksheet](source)
    a = wb[:_sheets]
    b = pop!(a)
    b[:title] = wsn
    wb[:_sheets] = vcat(b, a)
    ws = b
    datehead = reshape(data[:dates][data[:dates] .< o[:endofperiod]][1:o[:PCAdays]], 1, :)
    function openpyxlarraywriter(array::Array, refcell, ws)
        r, c = refcell
        for i in 1:size(array, 1), j in 1:size(array,2)
            ws[:cell](row = i+r-1, column = j+c-1, value = array[i,j])
        end
    end
    openpyxlarraywriter(hcat(datehead, datehead, datehead), (2, 2), ws)
    openpyxlarraywriter(imat, (3, 1), ws)
    ws[:freeze_panes] = "B3"
    wb[:save]("CRE Allocations.xlsx")
end

#dailycreroutine()


o = Dict()
o[:dataset] = :Smallset
o[:retrospectionperiod] = Month(6)
o[:numberofstocks] = 15
o[:PCAdays] = 4
o[:rankmode] = :Collective # {:Collective, :PEranks, :Spreadranks}
o[:addsecurities] = false
o[:pcatype] = :cov
o[:pcanorm] = false
o[:normtype] = 2 #{:nonorm, :meannorm, :zscore, :minmax} Note that if minmax is used using range[-1,1] or [0.1,0.9] is also an option need 2 take care of that.
o[:featurenum] = 8 #{1:NumberOfTradeables}
o[:lambda] = 0.40 #[0, +)
o[:churn] = 0.15 #[0.05, 0.3]
o[:initialstart] = Date(2013,1,1)
data = CustomDataInit(o)
PortfolioAllocations = portfolioconstructor(data, o)
quarterlyportfoliostats(data, o, PortfolioAllocations)

=#
#results2excel("Stockpicks", hcat(QuarterDates[qt], permutedims(hcat([data[:stocks][QuarterHoldings[i, :] .> 0] for i in 1:size(QuarterHoldings, 1)]...))))
#=
GridResults = Dict()
for lambda in round.(exp.(log(0.1):log(99.9999999999999)/50:log(10)), digits = 2), churn in collect(0.10:0.01:0.20), pcatype in [:cov, :cor], normtype in [0,1,2]
    o[:lambda], o[:churn], o[:pcatype], o[:normtype] = lambda, churn, pcatype, normtype
    PortfolioAllocations = portfolioconstructor(data, o)
    quarterlystats = quarterlyportfoliostats(data, o, PortfolioAllocations)
    GridResults[lambda, churn, pcatype, normtype] = quarterlystats
end


pcanorm = false
normtype = 2

Results2Print = Dict()
Results2Print[:Returns] = Array{Float64, 2}(undef, 51, 10)
for (i, lambda) in enumerate(round.(exp.(log(0.1):log(99.9999999999999)/50:log(10)), digits = 2)), (j, featurenum) in enumerate(collect(1:10))
    Results2Print[:Returns][i,j] = GridResults[lambda, featurenum, pcanorm, normtype][2, end]
end


results2excel("false2",Results2Print)
=#
