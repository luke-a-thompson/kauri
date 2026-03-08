# bench.jl — BenchmarkTools harness for CG2 vs EES25
#
# Uses @benchmarkable with a forced warmup + full sample suite so JIT
# compilation and GC noise don't pollute the numbers.

import Pkg
Pkg.activate(joinpath(@__DIR__), io = devnull)

using OrdinaryDiffEqLinear
using SciMLOperators
using LinearAlgebra
using BenchmarkTools
using Printf

include("ees25_method.jl")

# ---------------------------------------------------------------------------
# Problem (identical to cf_ees.jl)
# ---------------------------------------------------------------------------

function hat(v::Vector{Float64})::Matrix{Float64}
    return [  0.0   -v[3]  v[2]
              v[3]   0.0  -v[1]
             -v[2]   v[1]  0.0 ]
end

const ω₀ = [0.05, 0.30, 1.00]
const b   = [1.80, 0.60, 0.00]

function update_A!(A::Matrix{Float64}, u::Vector{Float64}, p, t)
    R = reshape(u, 3, 3)
    A .= kron(I(3), hat(ω₀ + R * b))
end

function make_prob()
    R₀ = Matrix{Float64}(I, 3, 3)
    u₀ = vec(R₀)
    A₀ = zeros(9, 9)
    op = MatrixOperator(A₀; update_func! = update_A!)
    return ODEProblem(op, u₀, (0.0, 5.0))
end

const PROB = make_prob()

solve_cg2()   = solve(PROB, CG2();   dt = 0.02, save_everystep = false, save_start = false, save_end = false)
solve_ees25() = solve(PROB, EES25(); dt = 0.02, save_everystep = false, save_start = false, save_end = false)

# Peak live-bytes helper — forces a full GC so the baseline is clean, runs f(),
# then reads gc_live_bytes() *without* a second GC.  The delta is the number of
# bytes that were freshly allocated and are still live when f() returns, i.e.
# the retained working set.  This works for tiny problems where the OS RSS
# never changes because everything fits inside Julia's pre-mapped heap.
function peak_live_delta_kib(f)::Int
    GC.gc(true)
    before = Base.gc_live_bytes()
    f()
    after = Base.gc_live_bytes()
    return div(max(after - before, 0), 1024)
end

# ---------------------------------------------------------------------------
# Warmup — ensures JIT compilation is done before we collect samples
# ---------------------------------------------------------------------------

print("Warming up CG2...   "); flush(stdout); solve_cg2();   println("done")
print("Warming up EES25... "); flush(stdout); solve_ees25(); println("done")

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

println("\nRunning benchmarks (this may take ~30 s)…")

suite = BenchmarkGroup()
suite["CG2"]   = @benchmarkable solve_cg2()
suite["EES25"] = @benchmarkable solve_ees25()

# Tune first so BenchmarkTools picks a sensible number of evaluations
tune!(suite)
results = run(suite; verbose = false)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

live_cg2   = peak_live_delta_kib(solve_cg2)
live_ees25 = peak_live_delta_kib(solve_ees25)

println("\n── Results ─────────────────────────────────────────────────────────")
for (name, t, live) in (("CG2", results["CG2"], live_cg2), ("EES25", results["EES25"], live_ees25))
    med = median(t)
    mn  = minimum(t)
    mx  = maximum(t)
    @printf("%-6s  median=%6.2f ms  min=%6.2f ms  max=%6.2f ms  allocs=%d  peak_live=%+d KiB\n",
            name,
            time(med) / 1e6,
            time(mn)  / 1e6,
            time(mx)  / 1e6,
            allocs(med),
            live)
end

r = BenchmarkTools.ratio(median(results["EES25"]), median(results["CG2"]))
@printf("\nEES25 / CG2  time ratio = %.2f×\n", time(r))
