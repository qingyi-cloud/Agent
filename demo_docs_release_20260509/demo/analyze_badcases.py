"""Generate a compact badcase report from simulation results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _md_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _money(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _action_name(row: dict[str, Any]) -> str:
    action = row.get("action") or {}
    return str(action.get("action", "") or "")


def _action_params(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("action") or {}
    params = action.get("params") or {}
    return params if isinstance(params, dict) else {}


def _summarize_actions(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    accepted_orders: list[str] = []
    income_ineligible_orders: list[str] = []
    wait_minutes = 0
    query_scan_minutes = 0
    for row in rows:
        action_name = _action_name(row)
        params = _action_params(row)
        query_scan_minutes += int(row.get("query_scan_cost_minutes", 0) or 0)
        if action_name == "wait":
            wait_minutes += int(params.get("duration_minutes", 0) or 0)
        if action_name == "take_order" and bool((row.get("result") or {}).get("accepted", False)):
            cargo_id = str(params.get("cargo_id", "") or "")
            accepted_orders.append(cargo_id)
            if not bool((row.get("result") or {}).get("income_eligible", True)):
                income_ineligible_orders.append(cargo_id)

    last = rows[-1] if rows else {}
    return {
        "steps": len(rows),
        "accepted_orders": accepted_orders,
        "wait_minutes": wait_minutes,
        "query_scan_minutes": query_scan_minutes,
        "income_ineligible_orders": income_ineligible_orders,
        "final_time": last.get("simulation_end_time", ""),
        "final_position": last.get("position_after", {}),
    }


def _bad_rules(driver: dict[str, Any]) -> list[dict[str, Any]]:
    rules = ((driver.get("preference_check") or {}).get("rules") or [])
    bad: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        penalty = float(rule.get("penalty", 0.0) or 0.0)
        if penalty > 0:
            bad.append(rule)
    return bad


def _format_bad_rule(rule: dict[str, Any]) -> str:
    name = str(rule.get("rule", "unknown"))
    penalty = _money(rule.get("penalty", 0.0))
    hints: list[str] = []
    for key in ("violations", "free_days", "off_days", "visit_days", "satisfied", "sequence_ok"):
        if key in rule:
            hints.append(f"{key}={rule[key]}")
    hint = f" ({', '.join(hints)})" if hints else ""
    return f"{name}: penalty={penalty}{hint}"


def build_report(monthly_income: dict[str, Any], run_summary: dict[str, Any], results_dir: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = monthly_income.get("summary") or {}
    drivers = monthly_income.get("drivers") or []
    driver_files = run_summary.get("driver_result_files") or {}

    lines: list[str] = [
        "# Badcase Report 202603",
        "",
        f"- generated_at: {generated_at}",
        f"- results_dir: `{results_dir}`",
        f"- simulation_duration_days: `{run_summary.get('simulation_duration_days', 'unknown')}`",
        f"- completed_steps: `{run_summary.get('completed_steps', 'unknown')}`",
        f"- total_net_income_all_drivers: `{_money(summary.get('total_net_income_all_drivers'))}`",
        f"- total_preference_penalty: `{_money(summary.get('total_preference_penalty'))}`",
        f"- failed_driver_count: `{summary.get('failed_driver_count', 0)}`",
        f"- total_token_usage: `{(summary.get('total_token_usage') or {}).get('total_tokens', 0)}`",
        "",
        "## Driver Badcase Overview",
        "",
        "| driver | net_income | preference_penalty | gross | cost | action_steps | accepted_orders | wait_minutes | scan_minutes | main_badcase |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    sorted_drivers = sorted(
        [d for d in drivers if isinstance(d, dict)],
        key=lambda d: (
            float((d.get("income") or {}).get("net_income", 0.0) or 0.0),
            str(d.get("driver_id", "")),
        ),
    )
    for driver in sorted_drivers:
        driver_id = str(driver.get("driver_id", ""))
        income = driver.get("income") or {}
        action_file = Path(str(driver_files.get(driver_id, "")))
        action_summary = _summarize_actions(action_file) if action_file else {}
        bad_rules = _bad_rules(driver)
        main_badcase = "; ".join(_format_bad_rule(rule) for rule in bad_rules[:3])
        if not main_badcase and float(income.get("net_income", 0.0) or 0.0) < 0:
            main_badcase = "negative net income without preference penalty"
        if driver.get("calculation_aborted"):
            main_badcase = f"calculation_aborted: {driver.get('validation_error')}"
        lines.append(
            "| {driver} | {net} | {penalty} | {gross} | {cost} | {steps} | {orders} | {wait} | {scan} | {badcase} |".format(
                driver=_md_escape(driver_id),
                net=_money(income.get("net_income")),
                penalty=_money(income.get("preference_penalty")),
                gross=_money(income.get("gross_income")),
                cost=_money(income.get("cost")),
                steps=action_summary.get("steps", 0),
                orders=len(action_summary.get("accepted_orders", [])),
                wait=action_summary.get("wait_minutes", 0),
                scan=action_summary.get("query_scan_minutes", 0),
                badcase=_md_escape(main_badcase or "none"),
            )
        )

    lines.extend(
        [
            "",
            "## Per-driver Details",
            "",
        ]
    )

    for driver in sorted_drivers:
        driver_id = str(driver.get("driver_id", ""))
        income = driver.get("income") or {}
        action_file = Path(str(driver_files.get(driver_id, "")))
        action_summary = _summarize_actions(action_file) if action_file else {}
        bad_rules = _bad_rules(driver)
        lines.extend(
            [
                f"### {driver_id}",
                "",
                f"- net_income: `{_money(income.get('net_income'))}`",
                f"- gross_income: `{_money(income.get('gross_income'))}`",
                f"- cost: `{_money(income.get('cost'))}`",
                f"- preference_penalty: `{_money(income.get('preference_penalty'))}`",
                f"- actions_file: `{action_file}`",
                f"- steps: `{action_summary.get('steps', 0)}`",
                f"- accepted_orders: `{', '.join(action_summary.get('accepted_orders', [])) or 'none'}`",
                f"- income_ineligible_orders: `{', '.join(action_summary.get('income_ineligible_orders', [])) or 'none'}`",
                f"- wait_minutes: `{action_summary.get('wait_minutes', 0)}`",
                f"- query_scan_minutes: `{action_summary.get('query_scan_minutes', 0)}`",
                f"- final_time: `{action_summary.get('final_time', '')}`",
                f"- final_position: `{json.dumps(action_summary.get('final_position', {}), ensure_ascii=False)}`",
            ]
        )
        if bad_rules:
            lines.append("- bad_rules:")
            for rule in bad_rules:
                lines.append(f"  - {_format_bad_rule(rule)}")
        else:
            lines.append("- bad_rules: `none`")
        if driver.get("calculation_aborted"):
            lines.append(f"- validation_error: `{driver.get('validation_error')}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(project_root: Path | None = None, results_dir: Path | None = None, output: Path | None = None) -> Path:
    layout_root = (project_root if project_root is not None else Path(__file__).resolve().parent).resolve()
    results_dir_path = (results_dir if results_dir is not None else layout_root / "results").resolve()
    monthly_income_file = results_dir_path / "monthly_income_202603.json"
    run_summary_file = results_dir_path / "run_summary_202603.json"
    output_file = (output if output is not None else results_dir_path / "badcase_report_202603.md").resolve()

    if not monthly_income_file.is_file():
        raise FileNotFoundError(f"missing monthly income file: {monthly_income_file}")
    if not run_summary_file.is_file():
        raise FileNotFoundError(f"missing run summary file: {run_summary_file}")

    report = build_report(_read_json(monthly_income_file), _read_json(run_summary_file), results_dir_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    print(str(output_file), flush=True)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a badcase report from monthly income and action logs.")
    parser.add_argument("--project-root", type=Path, default=None, help="Demo root containing results/.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Simulation results directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown path.")
    args = parser.parse_args()
    main(project_root=args.project_root, results_dir=args.results_dir, output=args.output)
