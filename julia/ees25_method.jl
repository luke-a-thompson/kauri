# ees25_method.jl — 3-stage Lie-group integrator (EES25)
#
# Stages (Kᵢ = h·A evaluated at the given point; ΔYᵢ are Lie-algebra elements):
#
#   K₁  = h·F(t,        Y₀)          ΔY₁ = K₁
#   Y₁  = exp(½ΔY₁) · Y₀
#   K₂  = h·F(t+½h,     Y₁)          ΔY₂ = -½ΔY₁ + K₂
#   Y₂  = exp(ΔY₂)  · Y₁
#   K₃  = h·F(t+h,      Y₂)          ΔY₃ = -2ΔY₂  + K₃
#   Y_{t+h} = exp(¼ΔY₃) · Y₂
#
# NOTE: exponential!() overwrites its first argument in-place.
# deepcopy() is used wherever a matrix must survive a subsequent exponential! call.

import OrdinaryDiffEqCore:
    OrdinaryDiffEqAlgorithm, OrdinaryDiffEqConstantCache,
    @cache, alg_cache, initialize!, perform_step!,
    _vec, increment_nf!

using OrdinaryDiffEqLinear: LinearMutableCache   # defined there, not in Core
using ExponentialUtilities: exponential!, ExpMethodGeneric, alloc_mem
using SciMLOperators: update_coefficients!
using LinearAlgebra: mul!, rmul!

# ── Algorithm ────────────────────────────────────────────────────────────────

"""
    EES25()

3-stage Lie-group integrator on SO(n).  The update rule is:

    ΔY₁ = hK₁,  Y₁ = exp(½ΔY₁)·Y₀
    ΔY₂ = -½ΔY₁ + hK₂,  Y₂ = exp(ΔY₂)·Y₁
    ΔY₃ = -2ΔY₂ + hK₃,  Y_{t+h} = exp(¼ΔY₃)·Y₂
"""
struct EES25 <: OrdinaryDiffEqAlgorithm end

# ── Mutable cache (in-place / array state) ───────────────────────────────────

@cache struct EES25Cache{uType, rateType, WType, expType} <: LinearMutableCache
    u::uType
    uprev::uType
    uprev2::uType
    tmp::uType
    fsalfirst::rateType
    W::WType
    Wtmp::WType
    k::rateType
    exp_cache::expType
end

function alg_cache(
        alg::EES25, u, rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
        uprev, uprev2, f, t, dt, reltol, p, calck,
        ::Val{true}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    W         = false .* _vec(rate_prototype) .* _vec(rate_prototype)'
    Wtmp      = similar(W)
    k         = zero(rate_prototype)
    fsalfirst = zero(rate_prototype)
    exp_cache = alloc_mem(f, ExpMethodGeneric())
    return EES25Cache(u, uprev, uprev2, zero(u), fsalfirst, W, Wtmp, k, exp_cache)
end

# ── Constant cache (out-of-place / scalar state) ─────────────────────────────

struct EES25ConstantCache <: OrdinaryDiffEqConstantCache end

function alg_cache(
        alg::EES25, u, rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
        uprev, uprev2, f, t, dt, reltol, p, calck,
        ::Val{false}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    return EES25ConstantCache()
end

# ── initialize! ──────────────────────────────────────────────────────────────

function initialize!(integrator, cache::EES25Cache)
    integrator.kshortsize = 2
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t)
    return increment_nf!(integrator.stats, 1)
end

# ── perform_step! ────────────────────────────────────────────────────────────

function perform_step!(integrator, cache::EES25Cache, repeat_step = false)
    (; t, dt, uprev, u, p) = integrator
    (; tmp, k, W, Wtmp, exp_cache) = cache
    exp_method = ExpMethodGeneric()
    L = integrator.f.f

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    update_coefficients!(L, uprev, p, t)
    copyto!(W, convert(AbstractMatrix, L))
    rmul!(W, dt)                           # W = K₁ = h·A(t, Y₀)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    # Y₁ = exp(½K₁)·Y₀, keeping W intact because it is reused in ΔY₂.
    copyto!(Wtmp, W)
    rmul!(Wtmp, 1 / 2)
    exponential!(Wtmp, exp_method, exp_cache)
    mul!(tmp, Wtmp, uprev)

    update_coefficients!(L, tmp, p, t + dt / 2)
    copyto!(Wtmp, convert(AbstractMatrix, L))
    rmul!(Wtmp, dt)                        # Wtmp = K₂ = h·A(t+½h, Y₁)
    @. W = Wtmp - (1 / 2) * W             # W = ΔY₂ = -½ΔY₁ + K₂

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    # Y₂ = exp(ΔY₂)·Y₁, using Wtmp as the exponential scratch buffer.
    copyto!(Wtmp, W)
    exponential!(Wtmp, exp_method, exp_cache)
    mul!(k, Wtmp, tmp)

    update_coefficients!(L, k, p, t + dt)
    copyto!(Wtmp, convert(AbstractMatrix, L))
    rmul!(Wtmp, dt)                        # Wtmp = K₃ = h·A(t+h, Y₂)
    @. W = Wtmp - 2 * W                   # W = ΔY₃ = -2ΔY₂ + K₃

    # ── Update ────────────────────────────────────────────────────────────────
    # Y_{t+h} = exp(¼ΔY₃)·Y₂, writing directly into the integrator state.
    copyto!(Wtmp, W)
    rmul!(Wtmp, 1 / 4)
    exponential!(Wtmp, exp_method, exp_cache)
    mul!(u, Wtmp, k)

    integrator.f(integrator.fsallast, u, p, t + dt)
    return increment_nf!(integrator.stats, 1)
end
