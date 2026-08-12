# E1–E4: What the experiments produce, and how to discuss them

All paths are relative to the repo root after running the runbook in README
("Revision Extensions"). Every table is mean ± std over 5 seeds, CIFAR-100,
Dirichlet α=0.5, N=10 sellers, buyer ratio 0.10, adv_rate 0.30, poison_rate
0.50, backdoor target class 1, ≤500 rounds with early stopping.

---

## E1 — Is the "Untenable Dilemma" an artifact of binary payment? (R2.O5, R3.O3)

**Artifacts**
- `analysis_partial/stepE1_tau_frontier.csv` — per-seed rows: `tau, seed, bsr, msr, adversary_revenue_share, rounds, acc, asr`
- `analysis_partial/stepE1_tau_frontier_agg.csv` — the headline table: one row per tau ∈ {0.0,…,0.9} with `msr_mean, bsr_mean, asr_mean, acc_mean, adversary_revenue_share_mean` (± std)
- Per-cell raw runs under `results/stepE1_tau_sweep_CIFAR100/…` (training logs, per-round selection)

**The two questions it answers**
1. *Continuous payment (E1a).* Compare `adversary_revenue_share` (continuous
   share of total payout captured by adversaries under weight-proportional
   payment) against binary MSR at the same tau. If adversaries still capture a
   revenue share ≈ their MSR (or higher), the dilemma is NOT an artifact of the
   binary R_i model — graded payments don't fix it because the filter's own
   weights are what leak value. If revenue share is much lower than MSR, the
   reviewer is partially right: binary accounting overstates the leak.
2. *Explicit selection-integrity objective (E1b).* The MSR-vs-Acc frontier over
   tau. Plot `msr_mean` (x) vs `acc_mean` (y). The paper's claim survives if NO
   tau achieves simultaneously low MSR (≲0.1) and high Acc (≳ the paper's
   no-defense accuracy band) — i.e., the frontier is a trade-off curve, not a
   free lunch. Expected shape: raising tau drives MSR down but drags BSR down
   with it (honest sellers rejected too), starving the model and lowering Acc;
   ASR should fall with MSR. If some tau DOES hit the joint objective, that's a
   positive result to report honestly: thresholded trust + explicit objective
   closes the gap, and the dilemma is about *objective design*, not capability.

---

## E3 — Empirical "Privacy Tax" (R1.O2)

**Artifacts**
- `analysis_partial/stepE3_dp_sweep.csv` / `stepE3_dp_sweep_agg.csv` — one row
  per (aggregator ∈ {fltrust, martfl}, dp_epsilon ∈ {none, 1, 4, 8}) with
  `bsr_mean` (PRIMARY), `msr_mean, asr_mean, acc_mean` (± std)

**The claim being tested** (Sec 7.1): distance/similarity-based filters
misclassify DP-induced variance in honest gradients as malicious.
Only honest sellers are noised (Gaussian mechanism, analytic Balle–Wang
calibration, δ=1e-5, clip C = the defense's tuned clip_norm 5.0); adversaries
keep white-box, noise-free uploads.

**Expected pattern**: BSR monotonically decreasing as ε decreases
(none → 8 → 4 → 1), i.e., more privacy ⇒ more honest rejection — the Privacy
Tax. A second-order effect worth discussing: MSR may *rise* at low ε because
noisy honest gradients make adversarial camouflage relatively easier (honest
cluster spreads out). If BSR is flat across ε, the claim is not substantiated
at these budgets — then check whether the tuned clip (C=5.0) already dominates
the noise (σ scales with C; report σ values from the logs) before concluding.

---

## E2 — The persistent-market premise: attrition + cumulative advantage (R2.O6, R2.O3)

**Artifacts (per run dir, e.g. `results/stepE2_multitask_fltrust_plain_CIFAR100/…/run_i_seed_s/`)**
- `multi_task_summary.csv` — one row per task t ∈ 1..5: `bsr, msr, acc, asr,
  n_active_benign, adversarial_fraction_active, cumulative_adversary_revenue,
  cumulative_benign_revenue, bandit_msr, payment_undefined_rounds`
- `multi_task_rounds.csv` — the same, per round (fine-grained attrition timing)
- `market_state.json` — final per-seller `{type, cumulative_payment, unpaid_streak, active}`
- `figures/attrition_curve.png` — active benign sellers vs task index
- `figures/cumulative_adversary_revenue.png` — adversary vs benign cumulative revenue
- `figures/market_composition.png` — adversarial fraction of active sellers vs task
- `figures/per_task_metrics.png` — BSR/MSR/Acc per task
- `figures/bandit_msr_per_task.png` — only for the `--bandit_persist` run

**The story these tell** (same sellers, rotating buyer query C_t, model reset
per task, weighted payments, benign exit after 3 unpaid rounds, adversaries
never exit):
1. *Attrition*: if the filter's false rejections are persistent (same honest
   sellers keep being unpaid under adverse selection), the attrition curve
   falls across tasks — single-task BSR understates the long-run damage
   because rejected sellers *leave*, and
2. *Composition*: the adversarial fraction of the active market rises
   mechanically — the marketplace selects FOR professional adversaries. This is
   the cross-task phenomenon no single FL run can show.
3. *Cumulative revenue*: adversary revenue compounds across tasks; compare the
   slope against benign revenue.
4. *Per-task BSR/MSR/Acc*: confirms single-task findings persist under query
   rotation (and shows whether martfl's baseline rotation behaves differently
   from fltrust across tasks).
5. *Bandit amortization* (fltrust_bandit config, run twice — with and without
   `--bandit_persist`): with persistence, `bandit_msr` should be LOW in task 1
   (exploration cost) and HIGH in tasks 2–5 (exploitation); without
   persistence, it stays at the task-1 level every task. The gap is the
   amortized exploration cost (supports R2.O3).

Counter-outcome to watch: if no benign seller ever hits 3 consecutive unpaid
rounds (BSR high and stable per round), attrition won't trigger — that itself
is a finding ("the filter pays honest sellers often enough to retain them"),
and exit_k can be tightened in the config to probe the margin.

---

## E4 — Is the valuation gap (Fig. 7) a KernelSHAP artifact? (R1.O3)

**Artifacts**
- `analysis_partial/stepE4_valuation_gap.csv` / `stepE4_valuation_gap_agg.csv` —
  one row per (defense ∈ {fedavg, fltrust, martfl, foolsgold},
  attack ∈ {no_attack, backdoor}, method ∈ {selection, kernelshap, banzhaf,
  leastcore, loo, influence}) with `benign_paid_pct, benign_discarded_pct,
  adv_paid_pct, adv_blocked_pct` (± std over seeds)
- Raw per-round scores in each run's `valuations.jsonl` (`kernelshap_score`,
  `banzhaf_score`, `leastcore_score`, `marginal_contrib_loo`, `influence_score`)

**How to read it**: the paper's gap = the filter pays/blocks sellers in a way
that disagrees with formal value. Because Banzhaf and Least Core are computed
from the SAME coalition/utility samples as KernelSHAP (and LOO/influence from
the same rounds), method-to-method differences are pure solution-concept
differences, not sampling noise. The claim is robust if the
benign-discarded / adversary-paid percentages tell the same qualitative story
across all five methods. Watch Least Core specifically: it's the
stability-oriented concept, so if the gap vanishes under Least Core but not
under the semivalues (Shapley/Banzhaf), that's a genuinely interesting
nuance — the gap would be about marginal-contribution accounting rather than
coalitional stability. Note `leastcore_score` may be absent for rounds where
the constraint-sampled LP was unbounded/infeasible (logged as
`leastcore_intractable`); the extractor skips those rounds for that method.

---

## One-slide summary for the meeting

| Exp | Reviewer point | Headline artifact | Claim survives if… |
|---|---|---|---|
| E1a | R2.O5/R3.O3 | `adversary_revenue_share` vs MSR | adversaries capture revenue ≈ MSR even under continuous payment |
| E1b | R2.O5/R3.O3 | tau frontier (`stepE1_tau_frontier_agg.csv`) | no tau gives low MSR AND high Acc simultaneously |
| E3 | R1.O2 | Privacy-Tax table (`stepE3_dp_sweep_agg.csv`) | BSR falls monotonically as ε ↓ (1 < 4 < 8 < none) |
| E2 | R2.O6/R2.O3 | attrition + composition + revenue curves | benign attrition ↓, adversarial fraction ↑, adversary revenue compounds; persistent bandit exploits after task 1 |
| E4 | R1.O3 | gap table across 5 methods (`stepE4_valuation_gap_agg.csv`) | benign-discarded/adversary-paid pattern is consistent across KernelSHAP, Banzhaf, Least Core, LOO, Influence |
