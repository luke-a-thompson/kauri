# order_test.jl — time-reversal defect test for EES25 and CG2
#
# Antisymmetric order p  ⟹  e_rev(h) = ‖Φ_{-h}(Φ_h(y₀)) − y₀‖ = O(h^{p+1})
# Expected slopes:
#   CG2   — antisymmetric order 2  →  slope 3
#   EES25 — antisymmetric order 5  →  slope 6
#
# Test ODE: dR/dt = L(R)·R on SO(3), L(R) = hat(ω₀ + R·b).
# Mixed space-frame (ω₀) and body-frame (R·b) — genuinely nonlinear and
# non-degenerate: all stage evaluations differ, exercising the full Lie-algebra
# order conditions.
#
# Backward step Φ_{-h}: same step function called with h < 0.
# Julia's exp(A::Matrix) allocates a fresh matrix and never touches A, so the
# standalone formulas are aliasing-free without any deepcopy.
#
# NOTE: Both methods are reimplemented as standalone 3-line functions rather than
# going through DifferentialEquations.jl, which does not natively support
# single-step backward integration (negative dt). The CG2 tableau is simple
# enough that the inline version is cleaner and safer.

import Pkg
Pkg.activate(joinpath(@__DIR__), io = devnull)

using LinearAlgebra, Statistics

# ── Lie algebra ──────────────────────────────────────────────────────────────

hat(v::AbstractVector) = [0.0 -v[3] v[2]; v[3] 0.0 -v[1]; -v[2] v[1] 0.0]

# ── Test ODE: dR/dt = L(R)·R ─────────────────────────────────────────────────
#
# L(R) = hat(ω₀ + R·b) — mixed space-frame (ω₀) and body-frame (R·b) velocity.
# This is non-degenerate: as R evolves, the body-frame contribution R·b rotates,
# so K₂ ≠ K₁ and ΔY₃ ≠ 0, exercising all nonlinear order conditions.
#
# Contrast: L(R) = hat(R·v) (pure body-frame) is DEGENERATE because
# exp(t·hat(v))·v = v, so all stages evaluate at the same angular velocity,
# EES25 reduces to the exact flow, and the reversal defect is identically zero.

const Ω0 = [0.1, 0.2, 0.3]   # fixed space-frame component
const B  = [0.5, -0.3, 0.1]  # body-frame coupling (not parallel to Ω0)
L(R) = hat(Ω0 + R * B)

# ── CG2 single step (Crouch–Grossman, 2nd order) ─────────────────────────────
#
#   K₁ = h·L(Y₀)
#   Y₁ = exp(K₁)·Y₀
#   K₂ = h·L(Y₁)
#   Y₊ = exp(½K₁)·exp(½K₂)·Y₀

function cg2_step(R, h::Float64)
    K1 = h * L(R)
    K2 = h * L(exp(K1) * R)
    return exp((1/2) * K1) * (exp((1/2) * K2) * R)
end

# ── EES25 single step (h < 0 gives the reversed step Φ_{-|h|}) ──────────────
#
#   K₁  = h·L(Y₀)          ΔY₁ = K₁
#   Y₁  = exp(½ΔY₁)·Y₀
#   K₂  = h·L(Y₁)          ΔY₂ = -½ΔY₁ + K₂
#   Y₂  = exp(ΔY₂)·Y₁
#   K₃  = h·L(Y₂)          ΔY₃ = -2ΔY₂  + K₃
#   Y₊  = exp(¼ΔY₃)·Y₂
#
# With h < 0 all Kᵢ flip sign; intermediate points Y₁,Y₂ move in the opposite
# direction; the combining coefficients (-½, -2, ¼) stay fixed.

function ees25_step(R, h::Float64)
    K1  = h * L(R)
    Y1  = exp((1/2) * K1) * R
    K2  = h * L(Y1)
    dY2 = -(1/2) * K1 + K2
    Y2  = exp(dY2) * Y1
    K3  = h * L(Y2)
    dY3 = -2 * dY2 + K3
    return exp((1/4) * dY3) * Y2
end

# ── Reversal-defect sweep ─────────────────────────────────────────────────────

R0 = exp(hat([0.8, 0.5, 0.3]))   # generic initial rotation (off any symmetry axis)
h0 = 0.5
m  = 20

"""Estimate the log₂ slope of e_rev(h) = ‖Φ_{-h}(Φ_h(R₀)) − R₀‖ vs h."""
function reversal_order(step::Function)
    errors = [norm(step(step(R0, h0 * 2.0^(-k)), -(h0 * 2.0^(-k))) - R0)
              for k in 0:m]
    slopes = [log(errors[k] / errors[k+1]) / log(2) for k in 1:m]

    # Collect slopes while errors decrease and stay above the roundoff floor
    stable = Float64[]
    for k in 1:m
        errors[k+1] < 1e-12 && break
        errors[k+1] > errors[k] && break
        k < 2 && continue   # skip first slope (may be pre-asymptotic)
        push!(stable, slopes[k])
    end
    return isempty(stable) ? mean(slopes[2:max(2, m÷2)]) : mean(stable)
end

p_cg2   = reversal_order(cg2_step)
p_ees25 = reversal_order(ees25_step)

println("CG2   reversal slope: ", round(p_cg2;   digits = 3), "  (expected 4)")
println("EES25 reversal slope: ", round(p_ees25; digits = 3), "  (expected 6)")
