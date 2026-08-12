import numpy as np

###############################################################################
# 1) Buyer Selection (Placeholder)
###############################################################################
def buyer_selection_with_scores(X_test, X_seller, costs, B, seed=42):
    """
    Placeholder function that:
      1) Computes a 'score' for each seller data point (mocked here by random).
      2) Selects points in descending order of score, subject to budget.

    Returns:
      selected_indices: list of chosen indices
      scores: np.array of shape (n,) with each data point's score
    """
    rng = np.random.RandomState(seed)

    n = X_seller.shape[0]
    # -----------------------------------------------------------
    # EXAMPLE "Score" Computation (Replace with your real logic!)
    # e.g. in a real scenario, you'd compute something like:
    #   score_j = sum over i in test of (x_i^T F^-1 x_j)^2
    # or partial derivatives from your selection objective
    # -----------------------------------------------------------
    # We'll just do random scores for demonstration:
    scores = rng.rand(n)

    # Sort data points by score/cost ratio (descending)
    score_per_cost = scores / (costs + 1e-9)
    sorted_inds = np.argsort(-score_per_cost)  # descending

    selected_indices = []
    total_cost = 0.0
    for idx in sorted_inds:
        if total_cost + costs[idx] <= B:
            selected_indices.append(idx)
            total_cost += costs[idx]
        else:
            break

    return selected_indices, scores


###############################################################################
# 2) Scenario A - Subset-Only Attack
###############################################################################
def miattack_scenarioA(final_subset,
                       selection_func,  # must replicate buyer's selection
                       X_seller, costs, B,
                       X_test_baseline,
                       candidate_samples):
    """
    Subset-Only Membership Inference Attack.

    Steps:
      For each candidate x^*:
        1) "In": Add x^* to X_test_baseline, re-run selection -> get subset_in
        2) "Out": Exclude x^*, re-run selection -> get subset_out
        3) Compare subset_in, subset_out to final_subset (the buyer's real selection)
           - If subset_in is closer => guess membership = True
           - Else => guess membership = False

    Args:
      final_subset      : list or set of indices the buyer ultimately chose
      selection_func    : function(X_test, X_seller, costs, B) -> (subset, scores)
                          We only need 'subset' here for scenario A.
      X_seller, costs, B: typical buyer selection inputs
      X_test_baseline   : a guess of the buyer's test set (without candidate)
      candidate_samples : list of np.array points [d-dim].

    Returns:
      dict: {candidate_index -> bool (True=member, False=non-member)}
    """
    final_subset_set = set(final_subset)

    membership_guesses = {}
    for i, x_star in enumerate(candidate_samples):
        # 1) Scenario "In"
        X_test_in = np.vstack([X_test_baseline, x_star[None, :]])
        subset_in, _ = selection_func(X_test_in, X_seller, costs, B)

        # 2) Scenario "Out"
        X_test_out = X_test_baseline
        subset_out, _ = selection_func(X_test_out, X_seller, costs, B)

        # 3) Compare to final subset
        set_in = set(subset_in)
        set_out = set(subset_out)

        dist_in = len(set_in ^ final_subset_set)   # symmetric difference
        dist_out = len(set_out ^ final_subset_set)

        membership_guesses[i] = (dist_in < dist_out)

    return membership_guesses


###############################################################################
# 3) Scenario B - Subset + Scores Attack
###############################################################################
def miattack_scenarioB(final_subset,
                       real_scores,      # array of shape (n,) containing buyer's actual computed scores
                       selection_func,   # must produce both subset + scoring
                       X_seller, costs, B,
                       X_test_baseline,
                       candidate_samples,
                       alpha=0.5):
    """
    Subset + Scores Membership Inference Attack.

    In addition to final_subset, the attacker also knows real_scores[j]
    for each seller datapoint j. This extra info helps refine the decision.

    Approach:
      For each candidate x^*:
        1) "In": Add x^*, re-run selection_func -> get (subset_in, scores_in)
        2) "Out": Exclude x^*, re-run selection_func -> get (subset_out, scores_out)
        3) Compare both 'subset_in' and 'scores_in' to the real subset/scores.
           We'll define a distance measure combining:
             - set distance of subsets
             - L1 distance in scores
           Weighted by alpha in [0,1].

           D_in  = alpha * sum_j|scores_in(j)-real_scores(j)| + (1-alpha)*|subset_in XOR final_subset|
           D_out = alpha * sum_j|scores_out(j)-real_scores(j)| + (1-alpha)*|subset_out XOR final_subset|
        4) If D_in < D_out => guess membership=True, else False.

    Args:
      final_subset  : list or set of indices actually chosen by buyer
      real_scores   : array of shape (n,), real scores the buyer assigned
      selection_func: function(X_test, X_seller, costs, B) -> (subset, scores)
      X_seller, costs, B: typical buyer selection inputs
      X_test_baseline  : guess of buyer's test set (w/o candidate)
      candidate_samples: list of np.array [d-dim]
      alpha: weighting factor for scores vs. subset distance

    Returns:
      dict: {candidate_index -> bool (membership guess)}
    """
    final_subset_set = set(final_subset)

    membership_guesses = {}
    for i, x_star in enumerate(candidate_samples):
        # 1) Scenario In
        X_test_in = np.vstack([X_test_baseline, x_star[None, :]])
        subset_in, scores_in = selection_func(X_test_in, X_seller, costs, B)
        set_in = set(subset_in)

        # 2) Scenario Out
        X_test_out = X_test_baseline
        subset_out, scores_out = selection_func(X_test_out, X_seller, costs, B)
        set_out = set(subset_out)

        # 3) Compute distances
        #   (A) Subset distance = size of symmetric difference
        dist_subset_in = len(set_in ^ final_subset_set)
        dist_subset_out = len(set_out ^ final_subset_set)

        #   (B) Scores distance = sum of absolute differences
        dist_scores_in = np.sum(np.abs(scores_in - real_scores))
        dist_scores_out = np.sum(np.abs(scores_out - real_scores))

        # Weighted combination
        D_in  = alpha * dist_scores_in  + (1-alpha) * dist_subset_in
        D_out = alpha * dist_scores_out + (1-alpha) * dist_subset_out

        membership_guesses[i] = (D_in < D_out)

    return membership_guesses


###############################################################################
# 4) DEMO / MAIN
###############################################################################
if __name__ == "__main__":
    np.random.seed(123)

    # --------------------------
    # (a) Generate Seller Data
    # --------------------------
    n = 10
    d = 3
    X_seller = np.random.randn(n, d)
    costs = 0.1 + 0.9*np.random.rand(n)
    B = 3.0  # small budget

    # --------------------------
    # (b) Generate Buyer Test Set
    # --------------------------
    m = 5
    X_test_buyer = np.random.randn(m, d)

    # --------------------------
    # (c) Buyer Does Selection
    # --------------------------
    final_subset, real_scores = buyer_selection_with_scores(
        X_test_buyer, X_seller, costs, B, seed=999
    )

    print("[Buyer] Final subset chosen:", final_subset)
    print("[Buyer] Real scores:", real_scores.round(3))

    # -------------------------------------------------
    # (d) Suppose we, the attacker, want to test
    #     membership for 2 candidate samples
    # -------------------------------------------------
    # We'll define two new random points as candidates:
    cand1 = np.random.randn(d)
    cand2 = np.random.randn(d)
    candidate_samples = [cand1, cand2]

    # We guess the baseline test set is exactly the real one (in practice might differ)
    X_test_baseline = X_test_buyer

    # -------------------------------------------------
    # (e) SCENARIO A: Subset-Only Attack
    # -------------------------------------------------
    # We define a small wrapper that calls buyer_selection_with_scores,
    # but ignore the 'scores' return since scenario A doesn't use it
    def subset_only_func(X_test, X_seller, costs, B):
        subset, _scores = buyer_selection_with_scores(X_test, X_seller, costs, B, seed=999)
        return subset, None

    scenarioA_guesses = miattack_scenarioA(
        final_subset=final_subset,
        selection_func=subset_only_func,
        X_seller=X_seller,
        costs=costs,
        B=B,
        X_test_baseline=X_test_baseline,
        candidate_samples=candidate_samples
    )
    print("\nScenario A (Subset-Only) Guesses:")
    for i, guess in scenarioA_guesses.items():
        print(f"  Candidate {i} => membership guess = {guess}")

    # -------------------------------------------------
    # (f) SCENARIO B: Subset + Scores Attack
    # -------------------------------------------------
    # We pass the original selection function that returns both subset & scores
    scenarioB_guesses = miattack_scenarioB(
        final_subset=final_subset,
        real_scores=real_scores,
        selection_func=buyer_selection_with_scores,
        X_seller=X_seller,
        costs=costs,
        B=B,
        X_test_baseline=X_test_baseline,
        candidate_samples=candidate_samples,
        alpha=0.5  # weigh subset difference & score difference equally
    )
    print("\nScenario B (Subset+Scores) Guesses:")
    for i, guess in scenarioB_guesses.items():
        print(f"  Candidate {i} => membership guess = {guess}")
