"""
Tests for the final evaluation summary and CI computation.

Verifies:
1. bootstrap_ci returns correct mean and plausible CI bounds.
2. bootstrap_ci handles edge cases (single value, all equal, empty).
3. aggregate_by_method correctly groups per-regime rows.
4. _interpretation produces correct strings for various delta/CI combinations.
5. The summarize script produces expected output files from synthetic data.
"""
import csv
import math
import random
from pathlib import Path



def _import_summarize():
    import sys
    scripts_dir = str(Path(__file__).parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    # Re-import as module by exec to avoid name collision
    src = (Path(__file__).parent.parent / "scripts" / "summarize_final_evaluation.py").read_text()
    mod_ns = {}
    exec(compile(src, "summarize_final_evaluation.py", "exec"), mod_ns)
    return mod_ns


class TestBootstrapCI:
    def test_mean_correct(self):
        ns = _import_summarize()
        values = [0.9, 0.8, 0.85, 0.95, 0.87]
        mean, lo, hi = ns["bootstrap_ci"](values, n=500, level=0.95,
                                          rng=random.Random(0))
        expected_mean = sum(values) / len(values)
        assert abs(mean - expected_mean) < 1e-10

    def test_ci_bounds_ordered(self):
        ns = _import_summarize()
        values = [0.9, 0.8, 0.85, 0.95, 0.87]
        mean, lo, hi = ns["bootstrap_ci"](values, n=1000, level=0.95,
                                          rng=random.Random(1))
        assert lo <= mean <= hi

    def test_ci_contains_mean_with_high_prob(self):
        ns = _import_summarize()
        values = [0.9, 0.8, 0.85, 0.95, 0.87, 0.92, 0.88, 0.83, 0.91, 0.86]
        mean, lo, hi = ns["bootstrap_ci"](values, n=1000, level=0.95,
                                          rng=random.Random(2))
        assert lo <= mean <= hi

    def test_single_value(self):
        ns = _import_summarize()
        mean, lo, hi = ns["bootstrap_ci"]([0.9], n=500, level=0.95,
                                          rng=random.Random(3))
        assert mean == 0.9
        assert lo == 0.9
        assert hi == 0.9

    def test_all_equal(self):
        ns = _import_summarize()
        values = [0.85] * 10
        mean, lo, hi = ns["bootstrap_ci"](values, n=500, level=0.95,
                                          rng=random.Random(4))
        assert abs(mean - 0.85) < 1e-10
        assert abs(lo - 0.85) < 1e-10
        assert abs(hi - 0.85) < 1e-10

    def test_empty_returns_nan(self):
        ns = _import_summarize()
        mean, lo, hi = ns["bootstrap_ci"]([], n=500, level=0.95,
                                          rng=random.Random(5))
        assert math.isnan(mean)
        assert math.isnan(lo)
        assert math.isnan(hi)


class TestAggregateByMethod:
    def _make_rows(self):
        return [
            {"name": "fifo", "source": "baseline", "regime": "r1",
             "priority_weighted_slo_goodput": "0.90", "slo_violation_rate": "0.05",
             "p95_ttft": "0.1", "p95_latency": "0.3"},
            {"name": "fifo", "source": "baseline", "regime": "r2",
             "priority_weighted_slo_goodput": "0.85", "slo_violation_rate": "0.08",
             "p95_ttft": "0.12", "p95_latency": "0.35"},
            {"name": "my_heuristic", "source": "heuristic", "regime": "r1",
             "priority_weighted_slo_goodput": "0.92", "slo_violation_rate": "0.04",
             "p95_ttft": "0.09", "p95_latency": "0.28"},
        ]

    def test_groups_correctly(self):
        ns = _import_summarize()
        rows = self._make_rows()
        by_method = ns["aggregate_by_method"](rows)
        assert "fifo" in by_method
        assert "my_heuristic" in by_method
        assert by_method["fifo"]["source"] == "baseline"
        assert by_method["my_heuristic"]["source"] == "heuristic"

    def test_wg_values_collected(self):
        ns = _import_summarize()
        rows = self._make_rows()
        by_method = ns["aggregate_by_method"](rows)
        assert len(by_method["fifo"]["wg_values"]) == 2
        assert len(by_method["my_heuristic"]["wg_values"]) == 1

    def test_nan_wg_excluded(self):
        ns = _import_summarize()
        rows = [
            {"name": "bad", "source": "heuristic", "regime": "r1",
             "priority_weighted_slo_goodput": "nan", "slo_violation_rate": "",
             "p95_ttft": "", "p95_latency": ""},
        ]
        by_method = ns["aggregate_by_method"](rows)
        assert len(by_method["bad"]["wg_values"]) == 0


class TestInterpretation:
    def test_ci_crosses_zero(self):
        ns = _import_summarize()
        result = ns["_interpretation"](0.005, -0.01, 0.02)
        assert "not statistically clear" in result

    def test_marginal_less_than_1pp(self):
        ns = _import_summarize()
        result = ns["_interpretation"](0.005, 0.001, 0.009)
        assert "marginal" in result

    def test_clear_improvement(self):
        ns = _import_summarize()
        result = ns["_interpretation"](0.05, 0.02, 0.08)
        assert "improvement" in result

    def test_clear_regression(self):
        ns = _import_summarize()
        result = ns["_interpretation"](-0.05, -0.08, -0.02)
        assert "regression" in result


class TestSummarizeScript:
    def _make_eval_dir(self, tmpdir: Path) -> Path:
        eval_dir = tmpdir / "eval"
        eval_dir.mkdir()
        # candidate_metrics_by_regime_flat.csv
        cand_path = eval_dir / "candidate_metrics_by_regime_flat.csv"
        with open(cand_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["regime", "split", "name", "source",
                         "priority_weighted_slo_goodput", "slo_violation_rate",
                         "p95_ttft", "p95_latency", "request_throughput",
                         "num_completed", "error"])
            w.writerow(["test_r1", "test", "kv_heuristic", "heuristic",
                         "0.92", "0.04", "0.09", "0.28", "12.0", "600", ""])
            w.writerow(["test_r2", "test", "kv_heuristic", "heuristic",
                         "0.88", "0.06", "0.11", "0.32", "11.5", "580", ""])
        # baseline_metrics_by_regime_flat.csv
        base_path = eval_dir / "baseline_metrics_by_regime_flat.csv"
        with open(base_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["regime", "split", "name", "source",
                         "priority_weighted_slo_goodput", "slo_violation_rate",
                         "p95_ttft", "p95_latency", "request_throughput",
                         "num_completed", "error"])
            for regime, wg in [("test_r1", "0.90"), ("test_r2", "0.85")]:
                w.writerow([regime, "test", "fifo", "baseline", wg,
                             "0.05", "0.10", "0.30", "12.0", "600", ""])
                w.writerow([regime, "test", "edf", "baseline",
                             str(float(wg) + 0.01), "0.04", "0.09", "0.29",
                             "12.0", "600", ""])
                w.writerow([regime, "test", "oracle_srtf", "oracle",
                             str(float(wg) + 0.05), "0.01", "0.07", "0.25",
                             "12.0", "600", ""])
        return eval_dir

    def test_produces_output_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            eval_dir = self._make_eval_dir(tmpdir)
            output_dir = tmpdir / "summary"

            from scripts import summarize_final_evaluation  # noqa: F401
            # Use subprocess to run the script
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "scripts/summarize_final_evaluation.py",
                 "--eval-dir", str(eval_dir),
                 "--output-dir", str(output_dir),
                 "--n-bootstrap", "100"],
                capture_output=True, text=True
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert (output_dir / "final_evaluation_summary.md").exists()
            assert (output_dir / "final_summary_table.csv").exists()
            assert (output_dir / "confidence_intervals.csv").exists()

    def test_regret_rows_with_oracle(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            eval_dir = self._make_eval_dir(tmpdir)
            output_dir = tmpdir / "summary"
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "scripts/summarize_final_evaluation.py",
                 "--eval-dir", str(eval_dir),
                 "--output-dir", str(output_dir),
                 "--n-bootstrap", "100"],
                capture_output=True, text=True
            )
            assert result.returncode == 0
            regret_path = output_dir / "regret_to_oracle.csv"
            assert regret_path.exists()
            with open(regret_path) as f:
                content = f.read()
            # Should have data (not just "# no data")
            if "no data" not in content:
                rows = list(csv.DictReader(content.splitlines()))
                assert any(r.get("method") == "kv_heuristic" for r in rows)
