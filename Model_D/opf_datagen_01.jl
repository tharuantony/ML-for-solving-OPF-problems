using Random, Distributions
using JSON
using Distributions: Normal, truncate
using ProgressMeter
using ArgParse
using MathOptInterface
using Dates
using JuMP
const MOI = MathOptInterface
using ThreadsX  # for parallel map
using PowerModels
using Ipopt

Random.seed!(123)
start_time01 = now()

# ------------------ Parse Arguments ------------------
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--netname", "-n"
            help = "The input network name"
            arg_type = String
            default = "nesta_case14_ieee"
        "--output", "-o"
            help = "The output name"
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

# ------------------ Helper: apply DNN warm-start to data & initial_guess ------------------
"""
Expect predictions_json to be a dict with optional keys:
  "gen" -> { gen_id => {"pg":..., "qg":..., "vg":... , "status":...} }
  "bus" -> { bus_id => {"va":..., "vm":...} }
  "branch" -> { branch_id => {"pf":..., "pt":..., "qf":..., "qt":...} }

IDs in predictions should match the keys in the PowerModels data (strings or ints).
This function:
 - writes *_start keys into the PowerModels data where applicable
 - builds an `initial_guess` dictionary ready for PowerModels.solve_opf(...)
"""
function apply_dnn_warmstart!(data::Dict, predictions::Dict)
    # Expect predictions keys: "pred-pg", "pg", "pred-qg", "qg", "pred-va", "va", "pred-vm", "vm",
    # "pf", "pt", "qf", "qt", "pred-pf", "pred-pt", "pred-qf", "pred-qt"

    # ---------------- GEN warm-start ----------------
    if haskey(predictions, "pg") || haskey(predictions, "pred-pg")
        for (gid, gpred) in merge(predictions.get("pg", Dict()), predictions.get("pred-pg", Dict()))
            sid = string(gid)
            if haskey(data["gen"], sid)
                gen = data["gen"][sid]
                gen["pg_start"] = gpred
            end
        end
    end
    if haskey(predictions, "qg") || haskey(predictions, "pred-qg")
        for (gid, gpred) in merge(predictions.get("qg", Dict()), predictions.get("pred-qg", Dict()))
            sid = string(gid)
            if haskey(data["gen"], sid)
                gen = data["gen"][sid]
                gen["qg_start"] = gpred
            end
        end
    end

    # ---------------- BUS warm-start ----------------
    for bus_key in ["va", "pred-va", "vm", "pred-vm"]
        if haskey(predictions, bus_key)
            for (bid, val) in predictions[bus_key]
                sid = string(bid)
                if haskey(data["bus"], sid)
                    bus = data["bus"][sid]
                    if bus_key in ["va", "pred-va"]; bus["va_start"] = val; end
                    if bus_key in ["vm", "pred-vm"]; bus["vm_start"] = val; end
                end
            end
        end
    end

    # ---------------- BRANCH warm-start ----------------
    for branch_key in ["pf", "pt", "qf", "qt", "pred-pf", "pred-pt", "pred-qf", "pred-qt"]
        if haskey(predictions, branch_key)
            for (bid, val) in predictions[branch_key]
                sid = string(bid)
                if haskey(data["branch"], sid)
                    branch = data["branch"][sid]
                    # Map predictions to branch fields
                    if branch_key in ["pf", "pred-pf"]; branch["pf_start"] = val; end
                    if branch_key in ["pt", "pred-pt"]; branch["pt_start"] = val; end
                    if branch_key in ["qf", "pred-qf"]; branch["qf_start"] = val; end
                    if branch_key in ["qt", "pred-qt"]; branch["qt_start"] = val; end
                end
            end
        end
    end

    return nothing
end



# ------------------ Scale loads ------------------
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

# ------------------ Main ------------------
args = parse_commandline()

# Prepare paths
data_path = "data/"
outdir = joinpath("C:/dnnwarmstart/data", "traindata01", args["netname"])
mkpath(outdir)
fileout = joinpath(outdir, args["output"] * ".json")
filein = joinpath("C:/dnnwarmstart/data/inputs", args["netname"] * ".m")
data = PowerModels.parse_file(filein)

# ------------------ Load DNN predictions and build warm-start --------------

predfile = joinpath("C:/dnnwarmstart/data/predictions", args["netname"], "dec-results.json")
predictions = open(predfile) do f
    JSON.parse(String(read(f)))
end
println("Loaded DNN predictions file: ", predfile)

# Build initial_guess using all predicted values and apply *_start into `data`
apply_dnn_warmstart!(data, predictions)

# Save the predictions-based warm-start for record
dnn_warm_file = joinpath(outdir, "dnn_warmstart.json")
open(dnn_warm_file, "w") do f
    write(f, JSON.json(predictions, 4))
end
println("Saved DNN warm start data to: ", dnn_warm_file)

# Load scaling parameters
Load_range = collect(args["lb"]:args["step"]:args["ub"])
nloads = length(data["load"])
total_runs = length(Load_range) * args["nperm"]

# Define solver with warm-start-friendly attributes
solver = Ipopt.Optimizer

# define solver options dictionary
solver_options = Dict(
    "print_level" => 0,
    "warm_start_init_point" => "yes",
    "mu_init" => 1e-6
)



# ------------------ Run AC OPF scenarios ------------------
@showprogress for _ in 1:1 end  # initialize progress bar
results = ThreadsX.map(1:total_runs) do idx
    run_start = now()
    µ_idx = div(idx - 1, args["nperm"]) + 1
    rep = mod(idx - 1, args["nperm"]) + 1
    µ = Load_range[µ_idx]
    σ = 0.01

    load_scale = get_load_coefficients_fast(µ, σ, nloads)
    newdata = scale_load(data, load_scale)

    # Re-apply detailed DNN warm-start to the scaled copy (newdata),
    # and build an initial_guess specific to this scenario.
    # Note: predictions are unchanged across scenarios in this example; if your DNN predicts per-scenario,
    # replace `predictions` with scenario-specific predictions here.
    apply_dnn_warmstart!(newdata, predictions)

    # Save warmstarted input occasionally (keeps parity with your original code)
    warmstart_file = joinpath(outdir, "newdata_with_warmstart.json")
    open(warmstart_file, "w") do f
        write(f, JSON.json(newdata, 4))
    end

    if idx % 50 == 0
        println("Running scenario $idx / $total_runs at $(Dates.now())")
    end

    # Solve AC OPF with the DNN initial guess
    # Pass `initial_guess = ig` into PowerModels.solve_opf
    opf_sol = PowerModels.solve_opf(
      newdata,
      ACPPowerModel,
      solver;
      setting = Dict(
        "output" => Dict("branch_flows" => true),
        "optimizer_options" => solver_options
      )
    )


    run_end = now()
    scenario_time = Millisecond(run_end - run_start).value / 1000  # seconds

    if opf_sol["termination_status"] == LOCALLY_SOLVED
        res = Dict{String, Any}()
        res["iter_id"] = idx
        res["scenario_time"] = scenario_time
        res["scale"] = mean(load_scale)
        res["pd"] = Dict(name => load["pd"] for (name, load) in newdata["load"])
        res["qd"] = Dict(name => load["qd"] for (name, load) in newdata["load"])
        res["vg"] = Dict(name => opf_sol["solution"]["bus"][string(gen["gen_bus"])]["vm"]
                         for (name, gen) in newdata["gen"] if data["gen"][name]["pmax"] > 0)
        res["pg"] = Dict(name => gen["pg"] for (name, gen) in opf_sol["solution"]["gen"]
                         if data["gen"][name]["pmax"] > 0)
        res["qg"] = Dict(name => gen["qg"] for (name, gen) in opf_sol["solution"]["gen"]
                         if data["gen"][name]["pmax"] > 0)
        res["pt"] = Dict(name => branch["pt"] for (name, branch) in opf_sol["solution"]["branch"])
        res["pf"] = Dict(name => branch["pf"] for (name, branch) in opf_sol["solution"]["branch"])
        res["qt"] = Dict(name => branch["qt"] for (name, branch) in opf_sol["solution"]["branch"])
        res["qf"] = Dict(name => branch["qf"] for (name, branch) in opf_sol["solution"]["branch"])
        res["va"] = Dict(name => bus["va"] for (name, bus) in opf_sol["solution"]["bus"])
        res["vm"] = Dict(name => bus["vm"] for (name, bus) in opf_sol["solution"]["bus"])
        res["objective"] = opf_sol["objective"]
        res["solve_time"] = opf_sol["solve_time"]
        # record that we used DNN warm-start
        #res["warmstart"] = "dnn"
        return res
    else
        return nothing
    end
end

end_time = now()
total_time = Millisecond(end_time - start_time01).value / 1000

# Filter successful OPF runs
res_stack = filter(!isnothing, results)
num_success = length(res_stack)
num_failed = total_runs - num_success

# ------------------ Extract constraints ------------------
pglim = Dict(name => (gen["pmin"], gen["pmax"]) for (name, gen) in data["gen"] if gen["pmax"] > 0)
qglim = Dict(name => (gen["qmin"], gen["qmax"]) for (name, gen) in data["gen"] if gen["pmax"] > 0)
vglim = Dict(name => (data["bus"][string(gen["gen_bus"])]["vmin"], data["bus"][string(gen["gen_bus"])]["vmax"])
               for (name, gen) in data["gen"] if gen["pmax"] > 0)
vm_lim = Dict(name => (bus["vmin"], bus["vmax"]) for (name, bus) in data["bus"])
rate_a = Dict(name => branch["rate_a"] for (name, branch) in data["branch"])
line_br_rx = Dict(name => (branch["br_r"], branch["br_x"]) for (name, branch) in data["branch"])
line_bg = Dict(name => (branch["g_to"] + branch["g_fr"], branch["b_to"] + branch["b_fr"]) for (name, branch) in data["branch"])

# ------------------ Save results ------------------
out_res = Dict{String, Any}()
out_res["experiments"] = res_stack
out_res["constraints"] = Dict("vg_lim" => vglim, "pg_lim" => pglim, "qg_lim" => qglim,
                              "vm_lim" => vm_lim, "rate_a" => rate_a,
                              "line_rx" => line_br_rx, "line_bg" => line_bg)

open(fileout, "w") do f
    write(f, JSON.json(out_res, 4))
    println("Saved output to: ", fileout)
end

end_time01 = now()
total_time01 = Millisecond(end_time01 - start_time01).value / 1000

println("----------------------------------------------------")
println("Total OPF runs: ", total_runs)
println("Converged (LOCALLY_SOLVED): ", num_success)
println("Failed/Other: ", num_failed)
println("Total AC OPF runtime: ", total_time, " seconds")
println("Total runtime of code: ", total_time01, " seconds")
println("----------------------------------------------------")
