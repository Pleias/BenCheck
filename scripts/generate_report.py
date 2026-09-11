#!/usr/bin/env python3
"""
Generate Markdown report from BenCheck experiment results.

Usage:
    python scripts/generate_report.py results/reasoning_datasets/ > report.md
    python scripts/generate_report.py results/reasoning_datasets/ -o report.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_experiment_summary(results_dir: Path) -> Dict[str, Any]:
    """Load experiment summary JSON."""
    summary_path = results_dir / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No experiment_summary.json found in {results_dir}")

    with open(summary_path) as f:
        return json.load(f)


def format_value(val: Any) -> str:
    """Format value with confidence interval if available."""
    if isinstance(val, dict) and "mean" in val:
        mean = val["mean"]
        if "ci_95_lower" in val and "ci_95_upper" in val:
            return f"{mean:.4f} [{val['ci_95_lower']:.4f}, {val['ci_95_upper']:.4f}]"
        elif "std" in val:
            return f"{mean:.4f} ± {val['std']:.4f}"
        else:
            return f"{mean:.4f}"
    elif isinstance(val, (int, float)):
        return f"{val:.4f}"
    else:
        return str(val)


def get_accuracy(check_data: Dict[str, Any]) -> str:
    """Extract accuracy from check data."""
    if "overall_accuracy" in check_data:
        return f"{check_data['overall_accuracy']:.4f}%"
    return "N/A"


def get_baseline_accuracy(context_req_data: Dict[str, Any], mode: str) -> str:
    """Extract baseline accuracy from context requirement check."""
    try:
        mode_data = context_req_data.get(mode, {})
        if "aggregated" in mode_data:
            agg = mode_data["aggregated"]
            if "baseline_accuracy" in agg:
                val = agg["baseline_accuracy"]
                if isinstance(val, dict) and "mean" in val:
                    return f"{val['mean']:.4f}%"
                elif isinstance(val, (int, float)):
                    return f"{val:.4f}%"
        return "N/A"
    except:
        return "N/A"


def generate_accuracy_table(datasets: Dict[str, Any]) -> str:
    """Generate accuracy summary table."""
    lines = ["## Accuracy Summary", ""]
    lines.append("| Dataset | Questions | Choices | Log-Likelihood | Generation | Enumeration Bias |")
    lines.append("|---------|-----------|---------|----------------|------------|------------------|")

    for dataset_name, dataset_data in datasets.items():
        n_questions = dataset_data.get("n_questions", "N/A")
        checks = dataset_data.get("checks", {})

        # Get number of choices from enumeration_bias
        n_choices = "N/A"
        enum_acc = "N/A"
        if "enumeration_bias" in checks:
            n_pos = checks["enumeration_bias"].get("n_positions", "N/A")
            n_choices = str(n_pos)
            enum_acc = get_accuracy(checks["enumeration_bias"])

        # Get accuracies from context_requirement
        log_acc = "N/A"
        gen_acc = "N/A"
        if "context_requirement" in checks:
            log_acc = get_baseline_accuracy(checks["context_requirement"], "log_likelihood")
            gen_acc = get_baseline_accuracy(checks["context_requirement"], "generation")

        lines.append(f"| {dataset_name} | {n_questions} | {n_choices} | {log_acc} | {gen_acc} | {enum_acc} |")

    lines.append("")
    lines.append("**Column Descriptions:**")
    lines.append("- **Log-Likelihood**: Accuracy using log-probability scoring (scores each continuation)")
    lines.append("- **Generation**: Accuracy using text generation (model generates answer letter)")
    lines.append("- **Enumeration Bias**: Accuracy from position bias check (should match log-likelihood)")
    lines.append("")
    return "\n".join(lines)


def generate_checks_overview(datasets: Dict[str, Any]) -> str:
    """Generate overview of all 5 checks."""
    lines = ["## Checks Overview", ""]
    lines.append("| Dataset | Length Bias | Position Bias | Context Req | None-of-Above | Grammar |")
    lines.append("|---------|-------------|---------------|-------------|---------------|---------|")

    for dataset_name, dataset_data in datasets.items():
        checks = dataset_data.get("checks", {})

        # Length bias
        length_status = "N/A"
        if "length_bias" in checks:
            effect = checks["length_bias"].get("effect_size_interpretation", "N/A")
            length_status = "✅ None" if effect == "negligible" else f"⚠️ {effect}"

        # Position bias (chi-square test on predictions) - NO ROUNDING!
        pos_status = "N/A"
        if "enumeration_bias" in checks:
            p_val = checks["enumeration_bias"].get("pred_chi2_p_value", 0)
            if isinstance(p_val, (int, float)):
                # Handle exact zero separately
                if p_val == 0.0:
                    pos_status = "⚠️ Biased (p<1e-10)"
                # Use scientific notation for very small p-values, full precision otherwise
                elif p_val < 0.0001:
                    pos_status = f"⚠️ Biased (p={p_val:.2e})"
                else:
                    pos_status = "✅ None" if p_val > 0.05 else f"⚠️ Biased (p={p_val:.6f})"

        # Context requirement
        context_status = "N/A"
        if "context_requirement" in checks:
            log_data = checks["context_requirement"].get("log_likelihood", {})
            if "aggregated" in log_data:
                agg = log_data["aggregated"]
                pct = agg.get("questions_requiring_context_pct")
                if isinstance(pct, (int, float)):
                    context_status = f"{pct:.4f}%"
                elif isinstance(pct, dict) and "mean" in pct:
                    context_status = f"{pct['mean']:.4f}%"

        # None of the above
        nota_status = "N/A"
        if "none_of_the_above" in checks:
            # Try new format first (placeholder_selected_rate)
            placeholder_rate = checks["none_of_the_above"].get("placeholder_selected_rate")
            if placeholder_rate is not None and isinstance(placeholder_rate, (int, float)):
                nota_status = f"{placeholder_rate:.4f}%"
            else:
                # Fall back to old format (accuracy_delta_pct) for backward compatibility
                delta = checks["none_of_the_above"].get("accuracy_delta_pct", 0)
                if isinstance(delta, (int, float)):
                    nota_status = f"{delta:.4f}%"

        # Grammar quality
        grammar_status = "N/A"
        if "grammar_quality" in checks:
            gq = checks["grammar_quality"]
            q_any = gq.get("Q_any_issue", {})
            q_any_rate = q_any.get("rate", 0) if isinstance(q_any, dict) else 0
            grammar_status = f"{q_any_rate*100:.4f}%"

        lines.append(f"| {dataset_name} | {length_status} | {pos_status} | {context_status} | {nota_status} | {grammar_status} |")

    lines.append("")
    lines.append("**Column Descriptions:**")
    lines.append("- **Length Bias**: Effect size (Cohen's d) - whether correct answers systematically differ in length")
    lines.append("- **Position Bias**: Chi-square test p-value on model predictions - tests if model prefers certain positions (p<0.05 = biased)")
    lines.append("- **Context Req**: % of questions requiring context to answer (lower = more answerable from choices alone)")
    lines.append("- **None-of-Above**: % of questions where model selected random placeholder (higher = model struggles to eliminate wrong answers)")
    lines.append("- **Grammar**: % of questions with grammar/spelling issues detected by GECToR")
    lines.append("")
    return "\n".join(lines)


def generate_detailed_check_sections(datasets: Dict[str, Any]) -> str:
    """Generate detailed sections for each check with all metrics."""
    lines = ["## Detailed Check Results", ""]

    for dataset_name, dataset_data in datasets.items():
        lines.append(f"### {dataset_name}")
        lines.append("")
        checks = dataset_data.get("checks", {})

        # 1. Length Bias Check
        if "length_bias" in checks:
            lb = checks["length_bias"]
            lines.append("**Length Bias Check:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            # Effect size and interpretation
            cohen_d = lb.get('cohen_d', 'N/A')
            if isinstance(cohen_d, (int, float)):
                lines.append(f"| Effect Size (Cohen's d) | {cohen_d:.6f} |")
            else:
                lines.append(f"| Effect Size (Cohen's d) | {cohen_d} |")
            lines.append(f"| Interpretation | {lb.get('effect_size_interpretation', 'N/A')} |")

            # Mean lengths
            mean_correct = lb.get('mean_correct_length', 'N/A')
            mean_incorrect = lb.get('mean_incorrect_length', 'N/A')
            if isinstance(mean_correct, (int, float)):
                lines.append(f"| Mean Correct Length | {mean_correct:.4f} words |")
            else:
                lines.append(f"| Mean Correct Length | {mean_correct} |")
            if isinstance(mean_incorrect, (int, float)):
                lines.append(f"| Mean Incorrect Length | {mean_incorrect:.4f} words |")
            else:
                lines.append(f"| Mean Incorrect Length | {mean_incorrect} |")

            # Standard deviations
            std_correct = lb.get('std_correct_length', 'N/A')
            std_incorrect = lb.get('std_incorrect_length', 'N/A')
            if isinstance(std_correct, (int, float)):
                lines.append(f"| Std Dev Correct Length | {std_correct:.4f} words |")
            else:
                lines.append(f"| Std Dev Correct Length | {std_correct} |")
            if isinstance(std_incorrect, (int, float)):
                lines.append(f"| Std Dev Incorrect Length | {std_incorrect:.4f} words |")
            else:
                lines.append(f"| Std Dev Incorrect Length | {std_incorrect} |")

            # Relative differences
            median_rel = lb.get('median_relative_length_diff', 'N/A')
            mean_rel = lb.get('mean_relative_length_diff', 'N/A')
            if isinstance(median_rel, (int, float)):
                lines.append(f"| Median Relative Diff | {median_rel:.6f} |")
            else:
                lines.append(f"| Median Relative Diff | {median_rel} |")
            if isinstance(mean_rel, (int, float)):
                lines.append(f"| Mean Relative Diff | {mean_rel:.6f} |")
            else:
                lines.append(f"| Mean Relative Diff | {mean_rel} |")

            # Percentage metrics
            correct_longest = lb.get('correct_is_longest_pct', 'N/A')
            correct_shortest = lb.get('correct_is_shortest_pct', 'N/A')
            if isinstance(correct_longest, (int, float)):
                lines.append(f"| Correct is Longest | {correct_longest:.2f}% |")
            else:
                lines.append(f"| Correct is Longest | {correct_longest} |")
            if isinstance(correct_shortest, (int, float)):
                lines.append(f"| Correct is Shortest | {correct_shortest:.2f}% |")
            else:
                lines.append(f"| Correct is Shortest | {correct_shortest} |")

            # Statistical tests
            mwu_stat = lb.get('mannwhitneyu_statistic', 'N/A')
            mwu_p = lb.get('mannwhitneyu_p_value', 'N/A')
            if isinstance(mwu_stat, (int, float)):
                lines.append(f"| Mann-Whitney U statistic | {mwu_stat:.2f} |")
            else:
                lines.append(f"| Mann-Whitney U statistic | {mwu_stat} |")
            if isinstance(mwu_p, (int, float)):
                lines.append(f"| Mann-Whitney U p-value | {mwu_p:.8f} |")
            else:
                lines.append(f"| Mann-Whitney U p-value | {mwu_p} |")

            rank_chi2_stat = lb.get('rank_chi2_statistic', 'N/A')
            rank_chi2_p = lb.get('rank_chi2_p_value', 'N/A')
            if isinstance(rank_chi2_stat, (int, float)):
                lines.append(f"| Rank χ² statistic | {rank_chi2_stat:.6f} |")
            else:
                lines.append(f"| Rank χ² statistic | {rank_chi2_stat} |")
            if isinstance(rank_chi2_p, (int, float)):
                lines.append(f"| Rank χ² p-value | {rank_chi2_p:.8f} |")
            else:
                lines.append(f"| Rank χ² p-value | {rank_chi2_p} |")

            # Length rank distribution
            rank_dist = lb.get('length_rank_distribution', 'N/A')
            if rank_dist != 'N/A' and isinstance(rank_dist, dict):
                rank_str = ', '.join(f"rank{k}: {v}" for k, v in sorted(rank_dist.items()))
                lines.append(f"| Length Rank Distribution | {rank_str} |")

            lines.append("")

        # 2. Enumeration Bias Check (Position Bias)
        if "enumeration_bias" in checks:
            eb = checks["enumeration_bias"]
            lines.append("**Enumeration Bias Check (Position Bias):**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            # Basic metrics
            n_positions = eb.get('n_positions', 'N/A')
            lines.append(f"| Number of Positions | {n_positions} |")

            overall_acc = eb.get('overall_accuracy', 'N/A')
            if isinstance(overall_acc, (int, float)):
                lines.append(f"| Overall Accuracy | {overall_acc:.4f}% |")
            else:
                lines.append(f"| Overall Accuracy | {overall_acc} |")

            # Gold position chi-square test
            gold_chi2_stat = eb.get('gold_chi2_stat', 'N/A')
            gold_chi2_p = eb.get('gold_chi2_p_value', 'N/A')
            if isinstance(gold_chi2_stat, (int, float)):
                lines.append(f"| Gold Position χ² statistic | {gold_chi2_stat:.6f} |")
            else:
                lines.append(f"| Gold Position χ² statistic | {gold_chi2_stat} |")
            if isinstance(gold_chi2_p, (int, float)):
                lines.append(f"| Gold Position χ² p-value | {gold_chi2_p:.8f} |")
            else:
                lines.append(f"| Gold Position χ² p-value | {gold_chi2_p} |")

            # Predicted position chi-square test
            pred_chi2_stat = eb.get('pred_chi2_stat', 'N/A')
            pred_chi2_p = eb.get('pred_chi2_p_value', 'N/A')
            if isinstance(pred_chi2_stat, (int, float)):
                lines.append(f"| Pred Position χ² statistic | {pred_chi2_stat:.6f} |")
            else:
                lines.append(f"| Pred Position χ² statistic | {pred_chi2_stat} |")
            if isinstance(pred_chi2_p, (int, float)):
                # Handle exact zero
                if pred_chi2_p == 0.0:
                    lines.append(f"| Pred Position χ² p-value | <1e-10 |")
                elif pred_chi2_p < 0.0001:
                    lines.append(f"| Pred Position χ² p-value | {pred_chi2_p:.2e} |")
                else:
                    lines.append(f"| Pred Position χ² p-value | {pred_chi2_p:.10f} |")
            else:
                lines.append(f"| Pred Position χ² p-value | {pred_chi2_p} |")

            # Accuracy by position
            acc_by_pos = eb.get("accuracy_by_gold_position", {})
            if acc_by_pos:
                acc_str = ', '.join(f'pos{k.split("_")[1]}: {v:.4f}%' for k, v in sorted(acc_by_pos.items()))
                lines.append(f"| Accuracy by Gold Position | {acc_str} |")

            # Distributions
            pred_dist = eb.get("pred_position_dist", [])
            gold_dist = eb.get("gold_position_dist", [])
            if pred_dist:
                lines.append(f"| Predicted Position Distribution | {pred_dist} |")
            if gold_dist:
                lines.append(f"| Gold Position Distribution | {gold_dist} |")
            lines.append("")

        # 3. Context Requirement Check
        if "context_requirement" in checks:
            cr = checks["context_requirement"]
            lines.append("**Context Requirement Check:**")
            lines.append("")

            for mode in ["log_likelihood", "generation"]:
                if mode in cr and "aggregated" in cr[mode]:
                    agg = cr[mode]["aggregated"]
                    lines.append(f"*{mode.replace('_', ' ').title()} Mode:*")
                    lines.append("")
                    lines.append("| Metric | Value |")
                    lines.append("|--------|-------|")

                    # Accuracy metrics
                    lines.append(f"| Baseline Accuracy | {format_value(agg.get('baseline_accuracy', 'N/A'))} |")
                    lines.append(f"| Empty Context Accuracy | {format_value(agg.get('empty_accuracy', 'N/A'))} |")
                    lines.append(f"| Lorem Ipsum Accuracy | {format_value(agg.get('lorem_accuracy', 'N/A'))} |")

                    # Accuracy drops
                    empty_drop = agg.get('empty_vs_baseline_drop', 'N/A')
                    lorem_drop = agg.get('lorem_vs_baseline_drop', 'N/A')
                    if isinstance(empty_drop, (int, float)):
                        lines.append(f"| Empty vs Baseline Drop | {empty_drop:.4f}% |")
                    elif isinstance(empty_drop, dict) and 'mean' in empty_drop:
                        lines.append(f"| Empty vs Baseline Drop | {format_value(empty_drop)} |")
                    else:
                        lines.append(f"| Empty vs Baseline Drop | {empty_drop} |")

                    if isinstance(lorem_drop, (int, float)):
                        lines.append(f"| Lorem vs Baseline Drop | {lorem_drop:.4f}% |")
                    elif isinstance(lorem_drop, dict) and 'mean' in lorem_drop:
                        lines.append(f"| Lorem vs Baseline Drop | {format_value(lorem_drop)} |")
                    else:
                        lines.append(f"| Lorem vs Baseline Drop | {lorem_drop} |")

                    # Context dependency metrics
                    lines.append(f"| Questions Requiring Context | {format_value(agg.get('questions_requiring_context_pct', 'N/A'))}% |")
                    lines.append(f"| Mean Context Dependency | {format_value(agg.get('mean_context_dependency', 'N/A'))} |")

                    # Statistical analysis summary
                    stat_analysis = agg.get('statistical_analysis', {})
                    if stat_analysis and isinstance(stat_analysis, dict):
                        # Show key statistical test results
                        for transform in ['empty', 'lorem']:
                            if transform in stat_analysis:
                                test_result = stat_analysis[transform]
                                test_name = test_result.get('test_name', 'unknown')
                                p_val = test_result.get('p_value', 'N/A')
                                significant = test_result.get('significant', 'N/A')
                                if isinstance(p_val, (int, float)):
                                    lines.append(f"| {transform.capitalize()} Test ({test_name}) | p={p_val:.6f}, sig={significant} |")

                    lines.append("")

        # 4. None of the Above Check
        if "none_of_the_above" in checks:
            nota = checks["none_of_the_above"]
            lines.append("**None of the Above Check:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            # Accuracy metrics
            baseline = nota.get('baseline_accuracy', 'N/A')
            baseline_str = f"{baseline:.4f}%" if isinstance(baseline, (int, float)) else baseline
            lines.append(f"| Baseline Accuracy | {baseline_str} |")

            baseline_count = nota.get('baseline_correct_count', 'N/A')
            lines.append(f"| Baseline Correct Count | {baseline_count} |")

            placeholder = nota.get('placeholder_accuracy', 'N/A')
            placeholder_str = f"{placeholder:.4f}%" if isinstance(placeholder, (int, float)) else placeholder
            lines.append(f"| Placeholder Accuracy | {placeholder_str} |")

            delta = nota.get('accuracy_delta_pct', 'N/A')
            delta_str = f"{delta:.4f}%" if isinstance(delta, (int, float)) else delta
            lines.append(f"| Accuracy Delta | {delta_str} |")

            # Selection rates
            placeholder_sel = nota.get('placeholder_selected_rate', 'N/A')
            placeholder_sel_str = f"{placeholder_sel:.4f}%" if isinstance(placeholder_sel, (int, float)) else placeholder_sel
            lines.append(f"| Placeholder Selected Rate | {placeholder_sel_str} |")

            placeholder_sel_count = nota.get('placeholder_selected_count', 'N/A')
            lines.append(f"| Placeholder Selected Count | {placeholder_sel_count} |")

            pred_changed = nota.get('prediction_changed_rate', 'N/A')
            pred_changed_str = f"{pred_changed:.4f}%" if isinstance(pred_changed, (int, float)) else pred_changed
            lines.append(f"| Prediction Changed Rate | {pred_changed_str} |")

            pred_changed_count = nota.get('prediction_changed_count', 'N/A')
            lines.append(f"| Prediction Changed Count | {pred_changed_count} |")

            # Transition metrics (GOOD - selects NOTA)
            cor_to_nota = nota.get('correct_to_placeholder_rate', 'N/A')
            cor_to_nota_str = f"{cor_to_nota:.4f}%" if isinstance(cor_to_nota, (int, float)) else cor_to_nota
            lines.append(f"| Correct→NOTA Rate | {cor_to_nota_str} |")

            cor_to_nota_count = nota.get('correct_to_placeholder_count', 'N/A')
            lines.append(f"| Correct→NOTA Count | {cor_to_nota_count} |")

            inc_to_nota = nota.get('incorrect_to_placeholder_rate', 'N/A')
            inc_to_nota_str = f"{inc_to_nota:.4f}%" if isinstance(inc_to_nota, (int, float)) else inc_to_nota
            lines.append(f"| Incorrect→NOTA Rate | {inc_to_nota_str} |")

            inc_to_nota_count = nota.get('incorrect_to_placeholder_count', 'N/A')
            lines.append(f"| Incorrect→NOTA Count | {inc_to_nota_count} |")

            # Error metrics (BAD - doesn't select NOTA)
            cor_to_wrong = nota.get('correct_to_wrong_rate', 'N/A')
            cor_to_wrong_str = f"{cor_to_wrong:.4f}%" if isinstance(cor_to_wrong, (int, float)) else cor_to_wrong
            lines.append(f"| Correct→Wrong Rate | {cor_to_wrong_str} |")

            cor_to_wrong_count = nota.get('correct_to_wrong_count', 'N/A')
            lines.append(f"| Correct→Wrong Count | {cor_to_wrong_count} |")

            inc_to_wrong = nota.get('incorrect_to_wrong_rate', 'N/A')
            inc_to_wrong_str = f"{inc_to_wrong:.4f}%" if isinstance(inc_to_wrong, (int, float)) else inc_to_wrong
            lines.append(f"| Incorrect→Wrong Rate | {inc_to_wrong_str} |")

            inc_to_wrong_count = nota.get('incorrect_to_wrong_count', 'N/A')
            lines.append(f"| Incorrect→Wrong Count | {inc_to_wrong_count} |")

            lines.append("")

        # 5. Grammar Quality Check
        if "grammar_quality" in checks:
            gq = checks["grammar_quality"]
            lines.append("**Grammar Quality Check:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            q_any = gq.get("Q_any_issue", {})
            q_any_rate = q_any.get("rate", 0) if isinstance(q_any, dict) else 0
            lines.append(f"| Questions with Any Issue | {q_any_rate*100:.4f}% |")

            q_prompt = gq.get("Q_prompt_issue", {})
            q_prompt_rate = q_prompt.get("rate", 0) if isinstance(q_prompt, dict) else 0
            lines.append(f"| Questions with Prompt Issue | {q_prompt_rate*100:.4f}% |")

            q_distractor = gq.get("Q_distractor_issue", {})
            q_distractor_rate = q_distractor.get("rate", 0) if isinstance(q_distractor, dict) else 0
            lines.append(f"| Questions with Distractor Issue | {q_distractor_rate*100:.4f}% |")

            q_helpful = gq.get("Q_helpful", {})
            q_helpful_rate = q_helpful.get("rate", 0) if isinstance(q_helpful, dict) else 0
            lines.append(f"| Helpful Cases (gold clean, distractors flagged) | {q_helpful_rate*100:.4f}% |")

            q_harmful = gq.get("Q_harmful", {})
            q_harmful_rate = q_harmful.get("rate", 0) if isinstance(q_harmful, dict) else 0
            lines.append(f"| Harmful Cases (gold flagged, distractors clean) | {q_harmful_rate*100:.4f}% |")

            gold_issue = gq.get("Q_gold_issue", {})
            gold_rate = gold_issue.get("rate", 0) if isinstance(gold_issue, dict) else 0
            lines.append(f"| Gold Answer Issues | {gold_rate*100:.4f}% |")

            flagged = gq.get("flagged_option_rate", {})
            flagged_rate = flagged.get("rate", 0) if isinstance(flagged, dict) else 0
            lines.append(f"| Flagged Option Rate | {flagged_rate*100:.4f}% |")

            distractor = gq.get("distractor_issue_rate", {})
            distractor_rate = distractor.get("rate", 0) if isinstance(distractor, dict) else 0
            lines.append(f"| Distractor Issue Rate | {distractor_rate*100:.4f}% |")

            # Grammar impact test (Fisher exact)
            grammar_test = gq.get("grammar_impact_test", {})
            if grammar_test:
                p_val = grammar_test.get("fisher_exact_p_value", "N/A")
                significance = grammar_test.get("significance", "N/A")
                if isinstance(p_val, (int, float)):
                    lines.append(f"| Grammar Impact Test (Fisher p-value) | {p_val:.6f} ({significance}) |")
                else:
                    lines.append(f"| Grammar Impact Test (Fisher p-value) | {p_val} ({significance}) |")

            # Option position flagged rate
            pos_flagged = gq.get("option_position_flagged_rate", {})
            if pos_flagged:
                pos_rates = [f"pos{k}: {v['rate']*100:.2f}%" for k, v in sorted(pos_flagged.items())]
                lines.append(f"| Flagged Rate by Position | {', '.join(pos_rates)} |")

            # Ambiguity histogram
            ambiguity_hist = gq.get("ambiguity_histogram", {})
            if ambiguity_hist:
                # Format as compact string: "0 flags: 2588, 1 flag: 210, ..."
                hist_str = ", ".join(f"{k}→{v}" for k, v in sorted(ambiguity_hist.items(), key=lambda x: int(x[0])))
                lines.append(f"| Ambiguity Histogram (flags→count) | {hist_str} |")

            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def generate_markdown_report(summary: Dict[str, Any]) -> str:
    """Generate complete Markdown report."""
    lines = ["# BenCheck Experiment Report", ""]

    # Experiment metadata
    config = summary.get("experiment_config", "N/A")
    timestamp = summary.get("timestamp", "N/A")
    lines.append(f"**Config:** `{config}`  ")
    lines.append(f"**Timestamp:** {timestamp}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Get datasets
    datasets = summary.get("datasets", {})

    # Accuracy table
    lines.append(generate_accuracy_table(datasets))
    lines.append("---")
    lines.append("")

    # Checks overview
    lines.append(generate_checks_overview(datasets))
    lines.append("---")
    lines.append("")

    # Detailed check sections
    lines.append(generate_detailed_check_sections(datasets))

    # Footer
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Accuracy**: Percentage of correct predictions")
    lines.append("- **Position Bias**: Chi-square test on model predictions (p > 0.05 = no bias)")
    lines.append("- **Context Requirement**: Percentage of questions that require context to answer")
    lines.append("- **None-of-Above**: Accuracy drop when adding placeholder distractor")
    lines.append("- **Grammar Issues**: Detected by GECToR grammar correction model")
    lines.append("- **Confidence Intervals**: 95% CI shown as [lower, upper] where available")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown report from BenCheck results"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to results directory (containing experiment_summary.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    # Load summary
    try:
        summary = load_experiment_summary(args.results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report = generate_markdown_report(summary)

    # Write output
    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
