using ExcelReaders

target = "UK"
countries = ["UK"] #Possible selections: "Australia", "Canada", "Japan", "New Zealand", "Norway", "Sweden", "Switzerland", "UK", "Eurozone". Note that USA is always included as all currency swaps are in terms of USD
variables = ["Equity Index"]#,"Interest Rates", "Government Bonds"]#["Equity Index", "Interest Rates", "Government Bonds"] # Possible selections: "Equity Index", "Interest Rates", "Government Bonds"
commodities = false

function initdatadaily(target, countries, variables, commodities)
    range = ["8", "4020"]
    dict = Dict{String, Array{String}}()
    dict["United States"] = ["B","E"]
    dict["Australia"] = ["G","J"]
    dict["Canada"] = ["L","O"]
    dict["Japan"] = ["Q","T"]
    dict["New Zealand"] = ["V","Y"]
    dict["Norway"] = ["AA","AD"]
    dict["Sweden"] = ["AF","AI"]
    dict["Switzerland"] = ["AK","AN"]
    dict["UK"] = ["AP","AS"]
    dict["Eurozone"] = ["AU","BD"]
    dict["Commodities"] = ["BF","BJ"]
    file = openxl("Daily Data.xlsx")
    inputs = Array{Float32}[]
    specs = Array{String}[]
    if commodities == true
        range[2] = "3248"
    end
    if "Government Bonds" in variables && ("Norway" in countries || "Sweden" in countries || "Switzerland" in countries)
        range[2] = "2934"
    end
    varrange = Int[]
    usvarrange = Int[]
    eurozonevarrange = Int[]
    varspec = Dict{Int, String}()
    usvarspec = Dict{Int, String}()
    eurozonevarspec = Dict{Int, String}()
    push!(varrange, 1)
    varspec[1] = "level"
    push!(eurozonevarrange, 1)
    eurozonevarspec[1] = "level"
    if "Equity Index" in variables
        push!(varrange, 2)
        push!(usvarrange, 1)
        push!(usvarrange, 2)
        push!(eurozonevarrange, 3)
        push!(eurozonevarrange, 5)
        push!(eurozonevarrange, 7)
        push!(eurozonevarrange, 9)
        varspec[2] = "level"
        usvarspec[1] = "level"
        usvarspec[2] = "level"
        eurozonevarspec[3] = "level"
        eurozonevarspec[5] = "level"
        eurozonevarspec[7] = "level"
        eurozonevarspec[9] = "level"
    end
    if "Government Bonds" in variables
        push!(varrange, 3)
        push!(usvarrange, 3)
        push!(eurozonevarrange, 4)
        push!(eurozonevarrange, 6)
        push!(eurozonevarrange, 8)
        push!(eurozonevarrange, 10)
        varspec[3] = "percent"
        usvarspec[3] = "percent"
        eurozonevarspec[4] = "percent"
        eurozonevarspec[6] = "percent"
        eurozonevarspec[8] = "percent"
        eurozonevarspec[10] = "percent"
    end
    if "Interest Rates" in variables
        push!(varrange, 4)
        push!(usvarrange, 4)
        push!(eurozonevarrange, 2)
        varspec[4] = "percent"
        usvarspec[4] = "percent"
        eurozonevarspec[2] = "percent"
    end
    sort!(varrange)
    sort!(usvarrange)
    sort!(eurozonevarrange)
    if commodities == true
        input = Array{Float32}(readxl(file, "Datasheet!" * dict["Commodities"][1] * range[1] * ":" * dict["Commodities"][2] * range[2]))
        push!(inputs, input)
        spec = Array{String}(size(input,2))
        spec .= "level"
        push!(specs, spec)
    end
    input = Array{Float32}(readxl(file, "Datasheet!" * dict["United States"][1] * range[1] * ":" * dict["United States"][2] * range[2]))[:,usvarrange]
    push!(inputs, input)
    push!(specs, map(x->usvarspec[x], usvarrange))

    if "Eurozone" in countries
        input = Array{Float32}(readxl(file, "Datasheet!" * dict["Eurozone"][1] * range[1] * ":" * dict["Eurozone"][2] * range[2]))[:,eurozonevarrange]
        push!(inputs, input)
        push!(specs, map(x->eurozonevarspec[x], eurozonevarrange))
        countries = filter!(x->x!="Eurozone",countries)
    end
    for country in countries
        input = Array{Float32}(readxl(file, "Datasheet!" * dict[country][1] * range[1] * ":" * dict[country][2] * range[2]))[:,varrange]
        push!(inputs, input)
        push!(specs, map(x->varspec[x], varrange))
    end
    output = Array{Float32}(readxl(file, "Datasheet!" * dict[target][1] * range[1] * ":" * dict[target][2] * range[2]))[:,1]
    output = output[end-1:-1:1]
    specs = vcat(specs...)
    inputs = hcat(inputs...)
    inputs = inputs[2:end,:]'
    inputs = inputs[:,end:-1:1]
    writedlm("inputs.txt", inputs)
    writedlm("specs.txt", specs)
    writedlm("ygolds.txt", output)
    return inputs, specs, output
end

inputs, specs, ygolds = initdatadaily(target, countries, variables, commodities)
println("Data Initilization is done.")
