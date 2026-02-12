using JuMP, Ipopt, PowerModels
using Random, Distributions
using JSON
using Distributions: Normal, truncate
using ProgressMeter
using ArgParse
using MathOptInterface
using Dates
import MathOptInterface
const MOI = MathOptInterface

using ThreadsX  # for parallel map

#PowerModels.silence()
Random.seed!(123)

""" Parse Arguments """
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--netname", "-n"
            help = "The input network name"
            arg_type = String
            default = "nesta_case14_ieee"
        "--output", "-o"
            help = "the output name"
            arg_type = String
            default = "traindata"
        "--lb"
            help = "The lb (in %) of the load interval"
            arg_type = Float64
            default = 0.8
        "--ub"
            help = "The ub (in %) of the load interval"
            arg_type = Float64
            default = 1.2
        "--step"
            help = "The step size resulting in a new load x + step"
            arg_type = Float64
            default = 0.1
        "--nperm"
            help = "The number of load permutations for each load scale"
            arg_type = Int
            default = 10
    end
    return parse_args(s)
end

function scale_load(data, scale_coef)
    newdata = deepcopy(data)
    for (i, (k, ld)) in enumerate(newdata["load"])
        if ld["pd"] > 0
            ld["pd"] *= scale_coef[i]
            ld["qd"] *= scale_coef[i]
        end
    end
    return newdata
end

function get_load_coefficients_fast(µ, σ, n)
    d = Truncated(Normal(µ, σ), µ - 0.1, µ + 0.1)
    x = rand(d, n)
    factor = µ * n / sum(x)
    return clamp.(x .* factor, 0.0, µ)
end

args = parse_commandline()

# Prepare paths
data_path = "data/"
outdir = joinpath("C:/dnn-opf/src/data", "traindata", args["netname"])
mkpath(outdir)
fileout = joinpath(outdir, args["output"] * ".json")
filein = joinpath("C:/dnn-opf/src/data/inputs", args["netname"] * ".m")
data = PowerModels.parse_file(filein)

Load_range = collect(args["lb"]:args["step"]:args["ub"])
nloads = length(data["load"])
total_runs = length(Load_range) * args["nperm"]

# Create solver backend once
solver = () -> begin
    opt = Ipopt.Optimizer()
    #MOI.set(opt, MOI.RawOptimizerAttribute("print_level"), 0)
    MOI.set(opt, MOI.RawOptimizerAttribute("print_level"), 4)
    return opt  # ✅ returns an MOI.ModelLike as required
end


start_time=now()
# Run OPF for each scenario in parallel
@showprogress for _ in 1:1 end  # to initialize ProgressMeter 
results = ThreadsX.map(1:total_runs) do idx
    µ_idx = div(idx - 1, args["nperm"]) + 1
    rep = mod(idx - 1, args["nperm"]) + 1
    µ = Load_range[µ_idx]
    ∑ = 0.01

    load_scale = get_load_coefficients_fast(µ, ∑, nloads)
    newdata = scale_load(data, load_scale)

    opf_sol = PowerModels.solve_opf(newdata, ACPPowerModel, solver; setting = Dict("output" => Dict("branch_flows" => true)))

    if opf_sol["termination_status"] == LOCALLY_SOLVED
        res = Dict{String, Any}()
        res["scale"] = mean(load_scale)
        res["pd"] = Dict(name => load["pd"] for (name, load) in newdata["load"])
        res["qd"] = Dict(name => load["qd"] for (name, load) in newdata["load"])
        res["vg"] = Dict(name => opf_sol["solution"]["bus"][string(gen["gen_bus"])]["vm"]
                         for (name, gen) in newdata["gen"]
                         if data["gen"][name]["pmax"] > 0)
        res["pg"] = Dict(name => gen["pg"] for (name, gen) in opf_sol["solution"]["gen"]
                         if data["gen"][name]["pmax"] > 0)
        res["qg"] = Dict(name => gen["qg"] for (name, gen) in opf_sol["solution"]["gen"]
                         if data["gen"][name]["pmax"] > 0)
        res["pt"] = Dict(name => data["pt"] for (name, data) in opf_sol["solution"]["branch"])
        res["pf"] = Dict(name => data["pf"] for (name, data) in opf_sol["solution"]["branch"])
        res["qt"] = Dict(name => data["qt"] for (name, data) in opf_sol["solution"]["branch"])
        res["qf"] = Dict(name => data["qf"] for (name, data) in opf_sol["solution"]["branch"])
        res["va"] = Dict(name => data["va"] for (name, data) in opf_sol["solution"]["bus"])
        res["vm"] = Dict(name => data["vm"] for (name, data) in opf_sol["solution"]["bus"])
        res["objective"] = opf_sol["objective"]
        res["solve_time"] = opf_sol["solve_time"]
        return res
    else
        return nothing
    end
end
end_time=now()
total_time = Millisecond(end_time - start_time).value / 1000
#filter successful opfs
res_stack = filter(!isnothing, results)
num_success = length(res_stack)
num_failed = total_runs - num_success

# Constraints extraction
pglim = Dict(name => (gen["pmin"], gen["pmax"]) for (name, gen) in data["gen"] if gen["pmax"] > 0)
qglim = Dict(name => (gen["qmin"], gen["qmax"]) for (name, gen) in data["gen"] if gen["pmax"] > 0)
vglim = Dict(name => (data["bus"][string(gen["gen_bus"])]["vmin"], data["bus"][string(gen["gen_bus"])]["vmax"])
               for (name, gen) in data["gen"] if gen["pmax"] > 0)
vm_lim = Dict(name => (bus["vmin"], bus["vmax"]) for (name, bus) in data["bus"])
rate_a = Dict(name => branch["rate_a"] for (name, branch) in data["branch"])
line_br_rx = Dict(name => (branch["br_r"], branch["br_x"]) for (name, branch) in data["branch"])
line_bg = Dict(name => (branch["g_to"] + branch["g_fr"], branch["b_to"] + branch["b_fr"]) for (name, branch) in data["branch"])

# Package output
out_res = Dict{String, Any}()
out_res["experiments"] = res_stack
out_res["constraints"] = Dict("vg_lim" => vglim, "pg_lim" => pglim, "qg_lim" => qglim,
                              "vm_lim" => vm_lim, "rate_a" => rate_a,
                              "line_rx" => line_br_rx, "line_bg" => line_bg)
#print detailed results
println("----------------------------------------------------")
println("Total OPF runs: ", total_runs)
println("Converged (LOCALLY_SOLVED): ", num_success)
println("Failed/Other: ", num_failed)
println("Total runtime: ", total_time, "seconds")
println("----------------------------------------------------")
# Write JSON output
open(fileout, "w") do f
    write(f, JSON.json(out_res, 4))
    println("Saved output to: ", fileout)
end
