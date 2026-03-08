import Pkg
Pkg.activate(joinpath(@__DIR__), io = devnull)

using ForwardDiff
using LinearAlgebra
using Printf
using CairoMakie

# grad_order_test.jl -- checkpointed vs reconstructed discrete-adjoint test
#
# We study a scalar parameter theta in the SO(3) dynamics
#     dR/dt = hat(omega0 + theta * R * b) * R,
# and the terminal loss
#     J(R(T)) = 0.5 * ||R(T) * e3 - target||^2.
#
# We compare two gradients for the same discrete method:
#   g_disc,chk(h)   = exact discrete adjoint using stored forward states
#   g_disc,recon(h) = memory-efficient discrete adjoint reconstructing each
#                     previous state with one negative step and no checkpoints
#
# The difference
#   e_recon(h) = |g_disc,recon(h) - g_disc,chk(h)|
# isolates the extra adjoint error caused by reconstruction and is the quantity
# that should reveal effective symmetry / antisymmetric order.

const T_FINAL = 5.0
const THETA0 = 1.0
const H0 = 0.5
const M = 7

const OMEGA0 = [0.05, 0.30, 1.00]
const B = [1.80, 0.60, 0.00]
const E3 = [0.0, 0.0, 1.0]
const TARGET = normalize([0.25, -0.55, 0.80])
const R0 = Matrix{Float64}(I, 3, 3)

function primal(x)
    if x isa ForwardDiff.Dual
        return ForwardDiff.value(x)
    end
    return x
end

function hat(v::AbstractVector{T})::Matrix{T} where {T <: Real}
    return T[
         0.0   -v[3]   v[2]
         v[3]   0.0   -v[1]
        -v[2]   v[1]   0.0
    ]
end

function so3_exp(xi::AbstractVector{T})::Matrix{T} where {T <: Real}
    K = hat(xi)
    theta2 = sum(abs2, xi)
    theta = sqrt(theta2)

    if abs(primal(theta)) < 1e-10
        a = one(T) - theta2 / 6 + theta2^2 / 120
        b = one(T) / 2 - theta2 / 24 + theta2^2 / 720
    else
        a = sin(theta) / theta
        b = (one(T) - cos(theta)) / theta2
    end

    return Matrix{T}(I, 3, 3) + a * K + b * (K * K)
end

function angular_velocity(R::AbstractMatrix{T}, theta::T)::Vector{T} where {T <: Real}
    return T.(OMEGA0) .+ theta .* (R * T.(B))
end

function cg2_step(R::AbstractMatrix{T}, h::Float64, theta::T)::Matrix{T} where {T <: Real}
    k1 = h .* angular_velocity(R, theta)
    y1 = so3_exp(k1) * R
    k2 = h .* angular_velocity(y1, theta)
    return so3_exp(0.5 .* k1) * (so3_exp(0.5 .* k2) * R)
end

function ees25_step(R::AbstractMatrix{T}, h::Float64, theta::T)::Matrix{T} where {T <: Real}
    k1 = h .* angular_velocity(R, theta)
    y1 = so3_exp(0.5 .* k1) * R

    k2 = h .* angular_velocity(y1, theta)
    dy2 = -0.5 .* k1 .+ k2
    y2 = so3_exp(dy2) * y1

    k3 = h .* angular_velocity(y2, theta)
    dy3 = -2.0 .* dy2 .+ k3
    return so3_exp(0.25 .* dy3) * y2
end

function forward_solve(step::Function, h::Float64, theta::T)::Matrix{T} where {T <: Real}
    nsteps = round(Int, T_FINAL / h)
    @assert isapprox(nsteps * h, T_FINAL; atol = 1e-12, rtol = 0.0)

    R = T.(R0)
    for _ in 1:nsteps
        R = step(R, h, theta)
    end
    return R
end

function terminal_cost(R::AbstractMatrix{T})::T where {T <: Real}
    spin = R * T.(E3)
    diff = spin - T.(TARGET)
    return dot(diff, diff) / 2
end

function terminal_cost_gradient(R::AbstractMatrix{Float64})::Vector{Float64}
    diff = R * E3 - TARGET
    return vec(diff * transpose(E3))
end

function local_pullback(
    step::Function,
    R_prev::AbstractMatrix{Float64},
    lambda_next::Vector{Float64},
    h::Float64,
    theta::Float64,
)::Tuple{Vector{Float64}, Float64}
    function scalar_map(z::AbstractVector{T}) where {T <: Real}
        R = reshape(z[1:9], 3, 3)
        theta_local = z[10]
        return dot(lambda_next, vec(step(R, h, theta_local)))
    end

    z = vcat(vec(R_prev), theta)
    grad = ForwardDiff.gradient(scalar_map, z)
    return Vector{Float64}(grad[1:9]), grad[10]
end

function forward_trajectory(step::Function, h::Float64, theta::Float64)::Vector{Matrix{Float64}}
    nsteps = round(Int, T_FINAL / h)
    trajectory = Vector{Matrix{Float64}}(undef, nsteps + 1)
    trajectory[1] = copy(R0)

    for n in 1:nsteps
        trajectory[n + 1] = Matrix{Float64}(step(trajectory[n], h, theta))
    end

    return trajectory
end

function adjoint_gradient_checkpointed(step::Function, h::Float64, theta::Float64)::Float64
    trajectory = forward_trajectory(step, h, theta)
    lambda = terminal_cost_gradient(trajectory[end])
    grad_theta = 0.0

    for n in length(trajectory)-1:-1:1
        lambda, local_grad_theta = local_pullback(step, trajectory[n], lambda, h, theta)
        grad_theta += local_grad_theta
    end

    return grad_theta
end

function adjoint_gradient_reconstructed(step::Function, h::Float64, theta::Float64)::Float64
    nsteps = round(Int, T_FINAL / h)
    R = Matrix{Float64}(forward_solve(step, h, theta))
    lambda = terminal_cost_gradient(R)
    grad_theta = 0.0

    for _ in nsteps:-1:1
        R_prev = Matrix{Float64}(step(R, -h, theta))
        lambda, local_grad_theta = local_pullback(step, R_prev, lambda, h, theta)
        grad_theta += local_grad_theta
        R = R_prev
    end

    return grad_theta
end

function fit_slope(h_values::Vector{Float64}, errors::Vector{Float64})::Float64
    valid = [k for k in eachindex(errors) if errors[k] > 1e-14]
    @assert length(valid) >= 2

    start_idx = max(first(valid), last(valid) - 3)
    sel = start_idx:last(valid)

    x = log.(h_values[sel])
    y = log.(errors[sel])
    x_mean = sum(x) / length(x)
    y_mean = sum(y) / length(y)
    return sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean) .^ 2)
end

function run_study(name::String, step::Function)::Tuple{Float64, Vector{Float64}, Vector{Float64}}
    h_values = [H0 * 2.0^(-k) for k in 0:M]
    recon_errors = Float64[]

    println()
    println(name)
    println("h              g_chk               g_recon             e_recon(h)")
    println("--------------------------------------------------------------------------------")

    for h in h_values
        g_chk = adjoint_gradient_checkpointed(step, h, THETA0)
        g_recon = adjoint_gradient_reconstructed(step, h, THETA0)
        e_recon = abs(g_recon - g_chk)
        push!(recon_errors, e_recon)
        @printf("%.8f   %+ .12e   %+ .12e   %.6e\n", h, g_chk, g_recon, e_recon)
    end

    slope = fit_slope(h_values, recon_errors)
    @printf("fitted reconstruction slope: %.3f\n", slope)
    return slope, h_values, recon_errors
end

function save_recon_error_plot(h_values::Vector{Float64}, errors::Vector{Float64}, label::String, outpath::String; color::Symbol = :coral)::Nothing
    fig = Figure(size = (400, 320))
    ax = Axis(fig[1, 1];
        xlabel = L"h",
        ylabel = L"e_{\mathrm{recon}}(h)",
        xscale = log10,
        yscale = log10,
        title = label,
    )
    scatterlines!(ax, h_values, errors; color = (color, 0.9), linewidth = 2, marker = :circle, markersize = 10)
    save(outpath, fig)
    return nothing
end

function main()::Nothing
    println("Checkpointed vs reconstructed discrete adjoint gradients")
    println("e_recon(h) = |g_disc,recon(h) - g_disc,chk(h)|")

    slope_cg2, h_cg2, err_cg2 = run_study("CG2", cg2_step)
    slope_ees25, h_ees25, err_ees25 = run_study("EES25", ees25_step)

    figures_dir = joinpath(@__DIR__, "figures")
    mkpath(figures_dir)
    save_recon_error_plot(h_cg2, err_cg2, "CG2", joinpath(figures_dir, "cg2_recon_error.pdf"); color = :coral)
    save_recon_error_plot(h_ees25, err_ees25, "EES25", joinpath(figures_dir, "ees25_recon_error.pdf"); color = :teal)
    println("\nSaved → $(joinpath(figures_dir, "cg2_recon_error.pdf"))")
    println("Saved → $(joinpath(figures_dir, "ees25_recon_error.pdf"))")

    println()
    @printf("Summary: CG2 slope = %.3f, EES25 slope = %.3f\n", slope_cg2, slope_ees25)
    return nothing
end

main()
