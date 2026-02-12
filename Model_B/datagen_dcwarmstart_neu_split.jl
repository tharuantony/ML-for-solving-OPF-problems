using JuMP, Ipopt, PowerModels
using Random, Distributions
using JSON
using ProgressMeter
using ArgParse
using Dates
using MathOptInterface
const MOI = MathOptInterface

PowerModels.silence()

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
            help = "Base output name (no extension). Writes <name>.jsonl and <name>_constraints.json"
            arg_type = String
            default = "traindata_neu_chunk"
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
        "--chunk"
            help = "Number of scenarios to run per chunk (e.g., 2000)"
            arg_type = Int
            default = 2000
    end
    return parse_args(s)
end

# ------------------ DC OPF warm-start ------------------
function run_dc_opf(filein::String)
    println("Running DC OPF warm start...")
    data = PowerModels.parse_file(filein)
    solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    dc_result = PowerModels.solve_opf(data, DCPPowerModel, solver)

    pg_init = Dict(k => dc_result["solution"]["gen"][k]["pg"] for k in keys(dc_result["solution"]["gen"]))
    va_init = Dict(k => dc_result["solution"]["bus"][k]["va"] for k in keys(dc_result["solution"]["bus"]))

    println("DC OPF completed. Warm-start data ready.")
    return pg_init, va_init
end

function apply_warm_start!(data, pg_init, va_init)
    for (gen_id, gen) in data["gen"]
        if haskey(pg_init, gen_id)
            gen["pg_start"] = pg_init[gen_id]
        end
    end
    for (bus_id, bus) in data["bus"]
        if haskey(va_init, bus_id)
            bus["va_start"] = va_init[bus_id]
            bus["vm_start"] = 1.0
        end
    end
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
outdir = joinpath("C:/dcwarm_start/data", "traindata", args["netname"])
mkpath(outdir)

# Output files (JSONL + constraints JSON)
fileout_jsonl = joinpath(outdir, args["output"] * ".jsonl")
fileout_constraints = joinpath(outdir, args["output"] * "_constraints.json")

filein = joinpath("C:/dcwarm_start/data/inputs", args["netname"] * ".m")
data = PowerModels.parse_file(filein)

# ------------------ DC warm-start ------------------
pg_init, va_init = run_dc_opf(filein)

# Save DC OPF warm-start separately (unchanged)
dcopf_out = Dict("pg_init" => pg_init, "va_init" => va_init)
dcopf_file = joinpath(outdir, "dcopf_warmstart_neu.json")
open(dcopf_file, "w") do f
    write(f, JSON.json(dcopf_out, 4))
end
println("Saved DC OPF warm start data to: ", dcopf_file)

# ------------------ AC OPF parameters ------------------
# safer range than collect(lb:step:ub) for tiny steps
n_steps = Int(round((args["ub"] - args["lb"]) / args["step"]))
Load_range = [args["lb"] + i * args["step"] for i in 0:n_steps]

nloads = length(data["load"])
scenarios = [(µ, rep) for µ in Load_range for rep in 1:args["nperm"]]
total_runs = length(scenarios)

solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
p = Progress(total_runs, 0.1)

println("Total scenarios: $total_runs")
println("Chunk size: $(args["chunk"])")
println("Appending experiments to: $fileout_jsonl")

# ------------------ Constraints (write once) ------------------
pglim = Dict(name => (gen["pmin"], gen["pmax"]) for (name, gen) in data["gen"] if gen["pmax"] > 0)
qglim = Dict(name => (gen["qmin"], gen["qmax"]) for (name, gen) in data["gen"] if gen["pmax"] > 0)
vglim = Dict(
    name => (data["bus"][string(gen["gen_bus"])]["vmin"], data["bus"][string(gen["gen_bus"])]["vmax"])
    for (name, gen) in data["gen"] if gen["pmax"] > 0
)
vm_lim = Dict(name => (bus["vmin"], bus["vmax"]) for (name, bus) in data["bus"])
rate_a = Dict(name => branch["rate_a"] for (name, branch) in data["branch"])
line_br_rx = Dict(name => (branch["br_r"], branch["br_x"]) for (name, branch) in data["branch"])
line_bg = Dict(name => (branch["g_to"] + branch["g_fr"], branch["b_to"] + branch["b_fr"]) for (name, branch) in data["branch"])

constraints_dict = Dict(
    "vg_lim" => vglim,
    "pg_lim" => pglim,
    "qg_lim" => qglim,
    "vm_lim" => vm_lim,
    "rate_a" => rate_a,
    "line_rx" => line_br_rx,
    "line_bg" => line_bg
)

open(fileout_constraints, "w") do f
    write(f, JSON.json(constraints_dict, 4))
end
println("Saved constraints to: ", fileout_constraints)

# ------------------ Chunked run + JSONL append ------------------
chunk_size = max(1, args["chunk"])

global total_success = 0
global total_failed = 0

open(fileout_jsonl, "a") do f
    global total_success, total_failed   # <-- ADD THIS LINE

    chunk_start = 1

    while chunk_start <= total_runs
        chunk_end = min(chunk_start + chunk_size - 1, total_runs)
        println("\nRunning chunk: $chunk_start:$chunk_end / $total_runs")

        n_success_chunk = 0
        n_failed_chunk = 0

        for idx in chunk_start:chunk_end
            µ, rep = scenarios[idx]
            σ = 0.01

            run_start = now()
            load_scale = get_load_coefficients_fast(µ, σ, nloads)
            newdata = scale_load(data, load_scale)
            apply_warm_start!(newdata, pg_init, va_init)

            ac_result = PowerModels.solve_opf(
                newdata,
                ACPPowerModel,
                solver;
                setting = Dict("output" => Dict("branch_flows" => true))
            )

            run_end = now()
            scenario_time = Millisecond(run_end - run_start).value / 1000

            if ac_result["termination_status"] == MOI.LOCALLY_SOLVED
                res = Dict{String, Any}()

                res["iter_id"] = idx
                res["scenario_time"] = scenario_time
                res["scale"] = mean(load_scale)

                res["pd"] = Dict(name => load["pd"] for (name, load) in newdata["load"])
                res["qd"] = Dict(name => load["qd"] for (name, load) in newdata["load"])

                res["vg"] = Dict(
                    name => ac_result["solution"]["bus"][string(gen["gen_bus"])]["vm"]
                    for (name, gen) in newdata["gen"]
                    if data["gen"][name]["pmax"] > 0
                )

                res["pg"] = Dict(
                    name => gen["pg"]
                    for (name, gen) in ac_result["solution"]["gen"]
                    if data["gen"][name]["pmax"] > 0
                )

                res["qg"] = Dict(
                    name => gen["qg"]
                    for (name, gen) in ac_result["solution"]["gen"]
                    if data["gen"][name]["pmax"] > 0
                )

                res["pt"] = Dict(name => br["pt"] for (name, br) in ac_result["solution"]["branch"])
                res["pf"] = Dict(name => br["pf"] for (name, br) in ac_result["solution"]["branch"])
                res["qt"] = Dict(name => br["qt"] for (name, br) in ac_result["solution"]["branch"])
                res["qf"] = Dict(name => br["qf"] for (name, br) in ac_result["solution"]["branch"])

                res["va"] = Dict(name => bus["va"] for (name, bus) in ac_result["solution"]["bus"])
                res["vm"] = Dict(name => bus["vm"] for (name, bus) in ac_result["solution"]["bus"])

                res["objective"] = ac_result["objective"]
                res["solve_time"] = ac_result["solve_time"]

                write(f, JSON.json(res))
                write(f, "\n")
                n_success_chunk += 1
            else
                n_failed_chunk += 1
            end

            next!(p)
        end

        flush(f)

        total_success += n_success_chunk
        total_failed += n_failed_chunk
        println("Chunk saved. Success: $n_success_chunk | Failed: $n_failed_chunk")

        GC.gc()
        chunk_start = chunk_end + 1
    end
end

# ------------------ Summary ------------------
end_time01 = now()
total_time01 = Millisecond(end_time01 - start_time01).value / 1000

println("----------------------------------------------------")
println("Total AC OPF runs attempted: ", total_runs)
println("Successful (LOCALLY_SOLVED): ", total_success)
println("Failed/other: ", total_failed)
println("Total runtime of code: ", total_time01, " seconds")
println("Output JSONL file: ", fileout_jsonl)
println("Constraints JSON file: ", fileout_constraints)
println("----------------------------------------------------")
