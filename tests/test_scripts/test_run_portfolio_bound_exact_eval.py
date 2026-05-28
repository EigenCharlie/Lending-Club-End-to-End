from __future__ import annotations

import pandas as pd

from scripts.search.run_portfolio_bound_exact_eval import (
    _build_pass2_candidate_ranks,
    _semantic_bucket,
)


def test_semantic_bucket_normalization() -> None:
    assert _semantic_bucket("return_global") == "return_global"
    assert _semantic_bucket("bound_proxy") == "bound_proxy"
    assert _semantic_bucket("family::tail_blended_uncertainty") == "family_guardrail"
    assert _semantic_bucket("forced::incumbent_region") == "forced_incumbent_region"
    assert _semantic_bucket("residual_ranked_fill") == "residual"


def test_pass2_selection_keeps_pass1_survivors_and_bucket_quota() -> None:
    shortlist = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "shortlist_bucket": "return_global",
                "realized_total_return": 100_000.0,
                "bound_proxy_rank": 1,
                "return_first_rank": 1,
            },
            {
                "candidate_rank": 2,
                "shortlist_bucket": "bound_proxy",
                "realized_total_return": 99_000.0,
                "bound_proxy_rank": 2,
                "return_first_rank": 2,
            },
            {
                "candidate_rank": 3,
                "shortlist_bucket": "family::tail_blended_uncertainty",
                "realized_total_return": 98_000.0,
                "bound_proxy_rank": 3,
                "return_first_rank": 3,
            },
            {
                "candidate_rank": 4,
                "shortlist_bucket": "conservative_region::blended_uncertainty",
                "realized_total_return": 97_000.0,
                "bound_proxy_rank": 4,
                "return_first_rank": 4,
            },
            {
                "candidate_rank": 5,
                "shortlist_bucket": "forced::incumbent_region",
                "realized_total_return": 96_000.0,
                "bound_proxy_rank": 5,
                "return_first_rank": 5,
            },
            {
                "candidate_rank": 6,
                "shortlist_bucket": "residual_ranked_fill",
                "realized_total_return": 95_000.0,
                "bound_proxy_rank": 6,
                "return_first_rank": 6,
            },
        ]
    )
    pass1_eval = pd.DataFrame(
        [
            {"candidate_rank": 1, "eval_random_state": 42, "all_bounds_hold": True},
            {"candidate_rank": 1, "eval_random_state": 84, "all_bounds_hold": True},
            {"candidate_rank": 2, "eval_random_state": 42, "all_bounds_hold": False},
            {"candidate_rank": 2, "eval_random_state": 84, "all_bounds_hold": False},
            {"candidate_rank": 3, "eval_random_state": 42, "all_bounds_hold": True},
            {"candidate_rank": 3, "eval_random_state": 84, "all_bounds_hold": True},
            {"candidate_rank": 4, "eval_random_state": 42, "all_bounds_hold": False},
            {"candidate_rank": 4, "eval_random_state": 84, "all_bounds_hold": False},
            {"candidate_rank": 5, "eval_random_state": 42, "all_bounds_hold": False},
            {"candidate_rank": 5, "eval_random_state": 84, "all_bounds_hold": False},
            {"candidate_rank": 6, "eval_random_state": 42, "all_bounds_hold": False},
            {"candidate_rank": 6, "eval_random_state": 84, "all_bounds_hold": False},
        ]
    )
    selected, summary = _build_pass2_candidate_ranks(
        shortlist=shortlist,
        pass1_eval=pass1_eval,
        bucket_min=1,
    )

    # Pass1 survivors (1 and 3) must always survive to pass2.
    assert 1 in selected
    assert 3 in selected
    # Bucket quota should expand selection beyond pass1 survivors.
    assert len(selected) >= 5
    assert summary["pass1_survivor_count"] == 2
