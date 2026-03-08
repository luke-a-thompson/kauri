# SO(3) ODE — CG2 vs EES25 Lie-group integrators, plotted on the unit sphere.
#
# State  : R ∈ SO(3), stored as u = vec(R) ∈ ℝ⁹
# ODE    : dR/dt = Ω(R) · R,  ω(R) = ω₀ + R·b  (state-dependent)
# Methods: CG2 (built-in) and EES25 (custom 3-stage)

import Pkg
Pkg.activate(joinpath(@__DIR__), io = devnull)

using OrdinaryDiffEqLinear
using SciMLOperators
using LinearAlgebra
using Printf
using CairoMakie

include("ees25_method.jl")

# ---------------------------------------------------------------------------
# Lie algebra
# ---------------------------------------------------------------------------

function hat(v::Vector{Float64})::Matrix{Float64}
    return [  0.0   -v[3]  v[2]
              v[3]   0.0  -v[1]
             -v[2]   v[1]  0.0 ]
end

# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

const ω₀ = [0.05, 0.30, 1.00]
const b   = [1.80, 0.60, 0.00]

function update_A!(A::Matrix{Float64}, u::Vector{Float64}, p, t)
    R = reshape(u, 3, 3)
    A .= kron(I(3), hat(ω₀ + R * b))
end

R₀    = Matrix{Float64}(I, 3, 3)
u₀    = vec(R₀)
A₀    = zeros(9, 9)
op    = MatrixOperator(A₀; update_func! = update_A!)
tspan = (0.0, 5.0)
prob  = ODEProblem(op, u₀, tspan)

# ---------------------------------------------------------------------------
# Solve + report
# ---------------------------------------------------------------------------

function run(alg, label)
    t0  = time()
    sol = solve(prob, alg; dt = 0.02, saveat = 0.04)
    elapsed = time() - t0

    so3_err = maximum(norm(reshape(u, 3, 3)' * reshape(u, 3, 3) - I(3)) for u in sol.u)

    @printf("%-6s  steps=%-4d  SO(3) err=%.2e  time=%.3f s\n",
            label, length(sol.t), so3_err, elapsed)
    return sol
end

sol_cg2   = run(CG2(),   "CG2")
sol_ees25 = run(EES25(), "EES25")

# ---------------------------------------------------------------------------
# Extract e₃ spin-axis trajectory for each solver
# ---------------------------------------------------------------------------

spin_axis(sol) = reduce(hcat, [Float32.(reshape(u, 3, 3)[:, 3]) for u in sol.u])'

traj_cg2   = spin_axis(sol_cg2)
traj_ees25 = spin_axis(sol_ees25)

# ---------------------------------------------------------------------------
# Unit sphere mesh
# ---------------------------------------------------------------------------

θ = range(0.0, π,   60)
φ = range(0.0, 2π, 120)
sx = Float32[sin(t) * cos(p) for t in θ, p in φ]
sy = Float32[sin(t) * sin(p) for t in θ, p in φ]
sz = Float32[cos(t)          for t in θ, p in φ]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

BG = RGBf(0.07, 0.07, 0.10)

fig = Figure(size = (1100, 950), backgroundcolor = BG)

ax = Axis3(
    fig[1, 1];
    aspect          = (1, 1, 1),
    title           = "SO(3) e₃ trajectory  ·  CG2 vs EES25",
    titlecolor      = :white,
    backgroundcolor = BG,
    xgridcolor      = RGBAf(1, 1, 1, 0.08),
    ygridcolor      = RGBAf(1, 1, 1, 0.08),
    zgridcolor      = RGBAf(1, 1, 1, 0.08),
    xticklabelcolor = :gray55,
    yticklabelcolor = :gray55,
    zticklabelcolor = :gray55,
)

# Sphere
surface!(ax, sx, sy, sz;
    color        = fill(RGBAf(0.30, 0.50, 0.85, 0.15), size(sx)...),
    shading      = NoShading,
    transparency = true,
)
wireframe!(ax, sx, sy, sz;
    color = RGBAf(0.4, 0.6, 1.0, 0.06), linewidth = 0.3)

# CG2 trajectory — coral/orange
lines!(ax, traj_cg2[:, 1], traj_cg2[:, 2], traj_cg2[:, 3];
    color = RGBAf(1.0, 0.45, 0.25, 0.9), linewidth = 2.2)

# EES25 trajectory — cyan/teal
lines!(ax, traj_ees25[:, 1], traj_ees25[:, 2], traj_ees25[:, 3];
    color = RGBAf(0.20, 0.85, 0.75, 0.9), linewidth = 2.2)

# Start markers (white dot, shared)
scatter!(ax, [traj_cg2[1, 1]], [traj_cg2[1, 2]], [traj_cg2[1, 3]];
    color = :white, markersize = 14, strokewidth = 0)

# End markers
scatter!(ax, [traj_cg2[end, 1]],   [traj_cg2[end, 2]],   [traj_cg2[end, 3]];
    color = RGBAf(1.0, 0.45, 0.25, 1.0), markersize = 12, strokewidth = 1,
    strokecolor = :white)
scatter!(ax, [traj_ees25[end, 1]], [traj_ees25[end, 2]], [traj_ees25[end, 3]];
    color = RGBAf(0.20, 0.85, 0.75, 1.0), markersize = 12, strokewidth = 1,
    strokecolor = :white)

# Legend
Legend(fig[1, 2],
    [LineElement(color = RGBAf(1.0, 0.45, 0.25, 1.0), linewidth = 3),
     LineElement(color = RGBAf(0.20, 0.85, 0.75, 1.0), linewidth = 3),
     MarkerElement(color = :white, marker = :circle, markersize = 10)],
    ["CG2", "EES25", "start (shared)"];
    backgroundcolor = RGBf(0.13, 0.13, 0.16),
    labelcolor = :white,
    framecolor = RGBAf(1, 1, 1, 0.1),
)

out = joinpath(@__DIR__, "so3_cg2.png")
save(out, fig; px_per_unit = 2)
println("Saved → $out")
