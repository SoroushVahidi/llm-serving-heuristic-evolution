import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.repair_swissai_v2_report import build_report


class SwissAIReportRepairTest(unittest.TestCase):
    def test_reconstructs_without_policy_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            combined = source / "combined"
            combined.mkdir(parents=True)
            windows = [f"w{i}" for i in range(2)]
            policies = ["fifo", "adaptive_chunked_prefill"]
            summary = pd.DataFrame({"window_id": windows, "source_file": ["a", "b"]})
            causal = pd.DataFrame({
                "window_id": windows,
                "feat_swiss_kv_proxy_p95": [10, 20],
                "feat_swiss_high_reuse_fraction": [0.3, 0.1],
                "feat_swiss_low_reuse_fraction": [0.1, 0.4],
                "feat_swiss_reuse_mean": [0.3, 0.1],
                "feat_swiss_reuse_p95": [0.4, 0.2],
                "feat_swiss_arrival_rate_1s": [10, 20],
                "feat_swiss_arrival_rate_5s": [10, 20],
                "feat_swiss_arrival_rate_20s": [10, 20],
                "feat_swiss_arrival_rate_60s": [10, 20],
                "feat_swiss_prompt_p95": [100, 200],
                "feat_swiss_output_p95": [20, 30],
                "feat_swiss_fraction_negative_slack": [0.0, 0.1],
                "feat_swiss_kv_pressure": [1, 2],
                "feat_swiss_token_budget_pressure": [0, 1],
            })
            rows = []
            for window in windows:
                for policy in policies:
                    rows.append({
                        "window_id": window,
                        "policy_name": policy,
                        "metric_arrival_normalized_weighted_goodput": 1.0,
                        "metric_completion_fraction": 1.0,
                        "metric_slo_violation_rate": 0.0,
                    })
            vectors = pd.DataFrame(rows)
            summary.to_csv(combined / "window_summary.csv", index=False)
            causal.to_csv(combined / "causal_features.csv", index=False)
            vectors.to_csv(combined / "policy_vectors.csv", index=False)
            # The production input is 512 x 27; this test targets the repair logic.
            import scripts.repair_swissai_v2_report as repair
            old = (repair.EXPECTED_WINDOWS, repair.EXPECTED_POLICIES, repair.NEW_POLICIES, repair.COVERAGE_FIELDS)
            repair.NEW_POLICIES = ["adaptive_chunked_prefill"]
            repair.COVERAGE_FIELDS = list(causal.columns[1:])
            repair.EXPECTED_WINDOWS = 2
            repair.EXPECTED_POLICIES = 2
            try:
                output = root / "reanalysis"
                result = build_report(source, output)
                self.assertEqual(result["matrix_integrity"]["cells"], 4)
                self.assertEqual(result["oracle"]["oracle_mean_anwg"], 1.0)
                self.assertFalse(result["full_policy_recomputation_required"])
                self.assertIn("synthetic", (output / "final_report.md").read_text())
                self.assertTrue((output / "input_integrity.json").exists())
            finally:
                repair.EXPECTED_WINDOWS, repair.EXPECTED_POLICIES, repair.NEW_POLICIES, repair.COVERAGE_FIELDS = old


if __name__ == "__main__":
    unittest.main()
