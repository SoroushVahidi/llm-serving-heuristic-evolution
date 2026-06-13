#!/usr/bin/env python3
"""
Load saved results and re-render tables without re-running experiments.

Usage:
    python scripts/build_baseline_tables.py --results-dir results/baseline_comparison/<timestamp>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from llmserveopt.evaluation.aggregate import make_summary_table, print_summary_table
from llmserveopt.plotting.tables import to_latex, to_markdown
from llmserveopt.plotting.figures import plot_all


def main():
    parser = argparse.ArgumentParser(description="Rebuild tables from saved results")
    parser.add_argument("--results-dir", required=True, help="Directory with per_run.csv")
    parser.add_argument("--latex", action="store_true", help="Also write LaTeX table")
    parser.add_argument("--figures", action="store_true", help="Regenerate figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "per_run.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    summary = make_summary_table(df)

    print("\nSummary Table:")
    print_summary_table(df)

    md_path = results_dir / "summary_table.md"
    to_markdown(summary, md_path)
    print(f"\nMarkdown table written to: {md_path}")

    if args.latex:
        tex_path = results_dir / "summary_table.tex"
        to_latex(summary, tex_path)
        print(f"LaTeX table written to: {tex_path}")

    if args.figures:
        fig_dir = results_dir / "figures"
        plot_all(df, fig_dir, summary_df=summary)
        print(f"Figures written to: {fig_dir}")


if __name__ == "__main__":
    main()
