module TestIPCW

using Test
using StableRNGs
using Random
using Distributions
using DataFrames
using CategoricalArrays
using MLJBase
using LogExpFunctions
using TMLE

include(joinpath(dirname(dirname(pathof(TMLE))), "test", "helper_fns.jl"))

"""
    test_ipcw_fixes_bias(build_estimator, Ψ, dataset, Ψ₀)

When the missingness is informative, the complete case analysis (`ipcw=false`, censored rows
simply dropped) is biased: its confidence interval misses the truth entirely. The IPCW estimator
recovers it on the very same dataset. `build_estimator(ipcw)` returns the estimator to run.
"""
function test_ipcw_fixes_bias(build_estimator, Ψ, dataset, Ψ₀)
    classic, _ = build_estimator(false)(Ψ, dataset; verbosity=0)
    lb, ub = confint(significance_test(classic))
    @test !(lb ≤ Ψ₀ ≤ ub)
    corrected, _ = build_estimator(true)(Ψ, dataset; verbosity=0)
    test_coverage(corrected, Ψ₀)
end

@testset "Censoring indicator utilities" begin
    # Test censoring_indicator_name
    @test TMLE.censoring_indicator_name(:Y) == :Δ_Y
    @test TMLE.censoring_indicator_name(:outcome) == :Δ_outcome

    # Test has_missing_outcomes
    df = DataFrame(Y = [1.0, 2.0, 3.0], X = [1, 2, 3])
    @test TMLE.has_missing_outcomes(df, :Y) == false
    allowmissing!(df, :Y)
    df.Y[2] = missing
    @test TMLE.has_missing_outcomes(df, :Y) == true
    @test TMLE.has_missing_outcomes(df, :X) == false

    # Test add_censoring_indicator
    df = DataFrame(Y = Union{Missing, Float64}[1.0, missing, 3.0, missing, 5.0])
    df = TMLE.add_censoring_indicator(df, :Y)
    @test hasproperty(df, :Δ_Y)
    Δ = df[!, :Δ_Y]
    @test Δ isa CategoricalVector
    @test unwrap.(Δ) == [1, 0, 1, 0, 1]

    # The censoring indicator must register as binary so nuisance model selection treats it as a
    # classification target (see `is_binary` in nuisance_estimators.jl).
    @test TMLE.is_binary(df, :Δ_Y)
end

"""
Generate a continuous outcome ATE problem with MAR missingness.
Missingness depends on W (confounders) creating a MAR pattern.

With `effect_modification=0` the effect is homogeneous and the missingness is ignorable for the
outcome model, so the complete case analysis is already consistent. Setting it non zero makes the
effect W dependent: dropping the censored rows then shifts the W distribution the counterfactual
means are averaged over, which biases the complete case analysis. The ATE stays 2 either way
since E[W] = 0.
"""
function ate_with_mar_missingness(;n=2000, effect_modification=0.)
    rng = StableRNG(42)
    W = randn(rng, n)
    T = rand(rng, n) .< LogExpFunctions.logistic.(0.5 .* W)
    # True ATE = 2
    Y = 2.0 .* T .+ W .+ effect_modification .* T .* W .+ 0.5 .* randn(rng, n)
    ATE_true = 2.0

    dataset = DataFrame(
        W = W,
        T = categorical(Int.(T)),
        Y = Vector{Union{Missing, Float64}}(Y)
    )
    # MAR missingness: P(missing | W) depends on W
    p_missing = LogExpFunctions.logistic.(-1.0 .+ 0.8 .* W)
    for i in 1:n
        if rand(rng) < p_missing[i]
            dataset.Y[i] = missing
        end
    end
    return dataset, ATE_true
end

@testset "IPCW OSE and TMLE" begin
    dataset, ATE_true = ate_with_mar_missingness(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    ose = Ose()
    result, _ = ose(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)

    tmle = Tmle()
    result, _ = tmle(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)

    # The effect is homogeneous and the missingness depends only on W, which the outcome model
    # adjusts for, so the complete case analysis is consistent here too. IPCW must not break it.
    result, _ = Tmle(ipcw=false)(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
    result, _ = Ose(ipcw=false)(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
end

@testset "Complete case is biased, IPCW is not: MAR missingness with effect modification" begin
    dataset, ATE_true = ate_with_mar_missingness(n=5000, effect_modification=2.5)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    test_ipcw_fixes_bias(ipcw -> Tmle(ipcw=ipcw), Ψ, dataset, ATE_true)
    test_ipcw_fixes_bias(ipcw -> Ose(ipcw=ipcw), Ψ, dataset, ATE_true)
end

"""
Generate a continuous outcome ATE problem with MAR missing *outcomes* AND MCAR missing
*covariates*. Only the former is corrected by IPCW; the latter exists to make the nuisance
fitting rows and the fluctuation rows two different row spaces, which is what exercises the
`align_to_rows` CV fold realignment.
"""
function ate_with_missing_covariates(;n=5000)
    rng = StableRNG(42)
    W = randn(rng, n)
    T = rand(rng, n) .< LogExpFunctions.logistic.(0.5 .* W)
    # Heterogeneous effect: 2 + 2.5W, so the ATE is 2 since E[W] = 0.
    Y = 2.0 .* T .+ W .+ 2.5 .* T .* W .+ 0.5 .* randn(rng, n)
    ATE_true = 2.0

    dataset = DataFrame(
        W = Vector{Union{Missing, Float64}}(W),
        T = categorical(Int.(T)),
        Y = Vector{Union{Missing, Float64}}(Y)
    )
    # MAR missing outcomes (depend on W), handled by IPCW.
    p_missing_Y = LogExpFunctions.logistic.(-1.0 .+ 0.8 .* W)
    for i in 1:n
        rand(rng) < p_missing_Y[i] && (dataset.Y[i] = missing)
        # MCAR missing covariate: nothing corrects for it, so the complete case deletion it
        # triggers must be innocuous for the test to mean anything. The retained rows keep the
        # same W distribution, hence the same ATE even though the effect is heterogeneous.
        rand(rng) < 0.25 && (dataset.W[i] = missing)
    end
    return dataset, ATE_true
end

@testset "CV-IPCW TMLE with missing covariates" begin
    dataset, ATE_true = ate_with_missing_covariates(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    test_ipcw_fixes_bias(ipcw -> Tmle(resampling=CV(nfolds=3), ipcw=ipcw), Ψ, dataset, ATE_true)
end

@testset "CV-IPCW OSE with missing covariates" begin
    dataset, ATE_true = ate_with_missing_covariates(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    test_ipcw_fixes_bias(ipcw -> Ose(resampling=CV(nfolds=3), ipcw=ipcw), Ψ, dataset, ATE_true)
end

function ate_with_treatment_dependent_missingness(;n=8000)
    rng = StableRNG(123)
    W = randn(rng, n)
    T = rand(rng, n) .< LogExpFunctions.logistic.(0.5 .* W)
    # Heterogeneous effect: 2 + 2.5W, so the ATE is 2 since E[W] = 0.
    Y = 2.0 .* T .+ W .+ 2.5 .* T .* W .+ 0.5 .* randn(rng, n)
    dataset = DataFrame(
        W = W,
        T = categorical(Int.(T)),
        Y = Vector{Union{Missing, Float64}}(Y)
    )
    p_missing = LogExpFunctions.logistic.(-0.5 .+ 1.2 .* W .+ 1.2 .* T)
    dataset[!, :Y][rand(rng, n) .< p_missing] .= missing
    return dataset, 2.0
end

@testset "Complete case is biased, IPCW is not: treatment dependent missingness" begin
    dataset, ATE_true = ate_with_treatment_dependent_missingness()
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    test_ipcw_fixes_bias(ipcw -> Tmle(ipcw=ipcw), Ψ, dataset, ATE_true)
    test_ipcw_fixes_bias(ipcw -> Ose(ipcw=ipcw), Ψ, dataset, ATE_true)
    # Severe downward bias, not a borderline miss: the truth is 2.
    classic, _ = Tmle(ipcw=false)(Ψ, dataset; verbosity=0)
    @test TMLE.estimate(classic) < 1
end

end

true
