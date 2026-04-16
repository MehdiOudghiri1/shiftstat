"""Example: discover slices with concentrated target failures."""

from __future__ import annotations

from shiftstat.audit import SliceDiscoverer
from shiftstat.datasets import make_hidden_subgroup_shift_classification


def run_example(random_state: int = 42) -> dict[str, object]:
    """Run failure-slice discovery on a synthetic hidden-failure benchmark."""

    data = make_hidden_subgroup_shift_classification(
        pattern="minority_subgroup_degradation",
        random_state=random_state,
    )
    discoverer = SliceDiscoverer(min_samples_leaf=25, random_state=random_state).fit(
        data.X_ref,
        data.y_ref,
        data.reference_predictions,
        data.X_target,
        data.y_target,
        data.target_predictions,
    )
    summary = discoverer.summary()
    top_slice = summary.iloc[0]
    return {
        "n_slices": int(len(summary)),
        "top_rule": str(top_slice["rule"]),
        "top_failure_share": float(top_slice["target_failure_share"]),
        "top_delta_error_rate": float(top_slice["delta_error_rate"]),
    }


if __name__ == "__main__":
    print(run_example())
