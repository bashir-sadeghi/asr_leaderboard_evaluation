"""Validate saved transcripts, find metric disagreements, and write an HTML report."""

from hashlib import sha256
from html import escape
from itertools import combinations
import json
from pathlib import Path

import numpy as np

TIE_TOLERANCE = 1e-6


def load_transcripts(paths):
    """Keep original metadata; require complete runs with identical IDs/references."""
    paths = [Path(path) for path in paths]
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Provide one or more distinct prediction files.")
    runs = []
    for index, path in enumerate(paths, 1):
        content = path.read_bytes()
        saved = json.loads(content)
        settings = saved.get("settings", {}) if isinstance(saved, dict) else {}
        rows = saved.get("predictions") if isinstance(saved, dict) else saved
        if not isinstance(settings, dict) or not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: expected a nonempty list of predictions.")
        records = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path}: each prediction must be an object.")
            clip_id = row.get("id")
            reference = row.get("reference", row.get("text"))
            prediction = row.get("prediction")
            if not isinstance(clip_id, str) or not clip_id or clip_id in records:
                raise ValueError(f"{path}: IDs must be unique, nonempty strings: {clip_id!r}.")
            if not isinstance(reference, str) or not isinstance(prediction, str):
                raise ValueError(f"{path}, {clip_id}: reference/text and prediction must be strings.")
            if "reference" in row and "text" in row and row["reference"] != row["text"]:
                raise ValueError(f"{path}, {clip_id}: reference and text disagree.")
            # Preserve meeting, duration, group counts, and batch provenance.
            records[clip_id] = {**row, "reference": reference, "text": reference}
        if "sample_ids" in settings:
            ids = settings["sample_ids"]
            if (not isinstance(ids, list) or not all(isinstance(x, str) for x in ids)
                    or len(ids) != len(records) or set(ids) != set(records)):
                raise ValueError(f"{path}: predictions must cover settings.sample_ids exactly.")
        if runs and {key: r["reference"] for key, r in records.items()} != {
                key: r["reference"] for key, r in runs[0]["records"].items()}:
            raise ValueError("Compare files with identical clip IDs and original references; none were dropped.")
        runs.append({"model": f"M{index}", "model_id": str(settings.get("model_id", path.stem)),
                     "file": str(path.resolve()), "sha256": sha256(content).hexdigest(),
                     "dataset_revision": settings.get("dataset_revision"), "records": records})
    revisions = {r["dataset_revision"] for r in runs if r["dataset_revision"] is not None}
    if len(revisions) > 1:
        raise ValueError("The input files declare different dataset revisions.")
    return runs


def find_disagreements(records, models, count):
    """Paired reversals for multiple models; percentile-rank gaps for one."""
    if count < 1:
        raise ValueError("Provide a positive number of examples.")
    by_model = {model: {r["id"]: r for r in records if r["model"] == model} for model in models}
    statistics, candidates = [], []
    if len(models) == 1:
        rows = [r for r in records if r["wer_percent"] is not None]
        ranks = []
        for metric in ("wer_percent", "semantic_distance"):
            # Average percentile ranks for ties; no comparison of incompatible units.
            values = [r[metric] for r in rows]
            if metric == "semantic_distance":
                values = np.round(values, 6)  # Suppress small floating-point differences.
            _, inverse, sizes = np.unique(values, return_inverse=True, return_counts=True)
            starts = np.cumsum(sizes) - sizes
            ranks.append((starts[inverse] + (sizes[inverse] - 1) / 2) / max(len(rows) - 1, 1))
        for row, wer_rank, semantic_rank in zip(rows, *ranks):
            gap = float(abs(wer_rank - semantic_rank))
            if gap > TIE_TOLERANCE:
                candidates.append({"id": row["id"], "models": models, "gap": gap,
                                   "wer_percentile": float(wer_rank), "semantic_percentile": float(semantic_rank),
                                   "kind": "different within-model percentile ranks"})
    else:
        for a, b in combinations(models, 2):
            stats = {"models": [a, b], "total": len(by_model[a]), "undefined_wer": 0,
                     "wer_ties": 0, "semantic_ties": 0, "both_ties": 0, "comparable": 0, "reversals": 0}
            for clip_id, left in by_model[a].items():
                right = by_model[b][clip_id]
                if left["wer_percent"] is None:
                    stats["undefined_wer"] += 1
                    continue
                dw = left["word_edits"]["total"] - right["word_edits"]["total"]
                ds = left["semantic_distance"] - right["semantic_distance"]
                wt, st = dw == 0, abs(ds) <= TIE_TOLERANCE
                stats["wer_ties"] += int(wt); stats["semantic_ties"] += int(st)
                stats["both_ties"] += int(wt and st)
                if not wt and not st:
                    stats["comparable"] += 1
                    if dw * ds < 0:
                        stats["reversals"] += 1
                        candidates.append({"id": clip_id, "models": [a, b], "gap": abs(ds),
                                           "wer_prefers": a if dw < 0 else b,
                                           "semantic_prefers": a if ds < 0 else b,
                                           "kind": "opposite model preferences"})
            statistics.append(stats)
    # Show distinct clips. A clip may still contribute several pairs to the counts.
    examples, used_ids = [], set()
    for candidate in sorted(candidates, key=lambda c: (-c["gap"], c["id"], c["models"])):
        if candidate["id"] not in used_ids:
            examples.append(candidate); used_ids.add(candidate["id"])
            if len(examples) == count:
                break
    return statistics, examples


def build_analysis(runs, records, settings, examples=5):
    """Build the report from already scored clips; do not run a text encoder again."""
    summaries = []
    for run in runs:
        rows = [r for r in records if r["model"] == run["model"]]
        totals = {
            kind: {key: sum(r[kind][key] for r in rows)
                   for key in ("ins", "del", "sub", "total", "ref_len")}
            for kind in ("word_edits", "character_edits")
        }
        words, chars = totals["word_edits"], totals["character_edits"]
        summaries.append({
            "model": run["model"], "clips": len(rows), **totals,
            "wer_percent": 100 * words["total"] / words["ref_len"] if words["ref_len"] else None,
            "cer_percent": 100 * chars["total"] / chars["ref_len"] if chars["ref_len"] else None,
            "semantic_distance": float(np.mean([r["semantic_distance"] for r in rows])),
            "empty_normalized_references": sum(r["wer_percent"] is None for r in rows),
        })
    pairwise, selected = find_disagreements(records, [r["model"] for r in runs], examples)
    rule = ("Strict paired WER/semantic preference reversals, largest semantic gap first; one example per clip."
            if len(runs) > 1 else
            "Largest within-model WER/semantic percentile gaps; average ranks for ties, semantic values rounded to six decimals.")
    return {
        "settings": {**settings, "selection_rule": rule, "max_examples": examples,
                     "paired_semantic_tie_tolerance": TIE_TOLERANCE,
                     "single_model_semantic_rank_decimals": 6},
        "sources": [{k: v for k, v in run.items() if k != "records"} for run in runs],
        "summary": summaries, "pairwise": pairwise, "examples": selected, "records": records,
    }


def write_report(result, path):
    """Write a standalone, escaped HTML report using only the supplied scores."""
    path = Path(path)
    if path.suffix.lower() != ".html":
        raise ValueError("The transcript report must use an .html output path.")

    def table(headers, rows):
        head = "".join("<th scope='col'>" + escape(str(x)) + "</th>" for x in headers)
        body = "".join("<tr>" + "".join("<td>" + escape(str(x)) + "</td>" for x in row)
                       + "</tr>" for row in rows)
        return "<div class='table-wrap'><table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table></div>"

    def metric_row(row):
        rates = ["N/A" if row[key] is None else f"{row[key]:.2f}%"
                 for key in ("wer_percent", "cer_percent")]
        return [row["model"], *rates, f"{row['semantic_distance']:.3f}"]

    settings = result["settings"]
    parts = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Metric disagreement report</title>
<style>
body{font:16px/1.55 system-ui;color:#182f3e;max-width:1050px;margin:36px auto;padding:0 24px}
h1,h2{line-height:1.25}h2{margin-top:30px}a{color:#1b6395}code{overflow-wrap:anywhere}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}
th,td{border-bottom:1px solid #d6e1e7;padding:9px 12px;text-align:left;vertical-align:top}
th{background:#edf3f7}td{font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.note{color:#52646f;font-size:14px}.callout{background:#f1f6fa;border-left:3px solid #2166a5;padding:12px 16px}
details{border:1px solid #d6e1e7;border-radius:6px;margin:14px 0}summary{cursor:pointer;font-weight:600;padding:12px}
.detail-body{padding:0 16px 12px}.clip{font:12px/1.5 monospace;overflow-wrap:anywhere}
@media print{body{font-size:10pt;margin:0}details{break-inside:avoid}.table-wrap{overflow:visible}table{font-size:9pt}}
</style></head><body><h1>Metric disagreement report</h1>"""]
    parts.append("<p class='note'>Scores recorded " + escape(str(settings.get("created_utc", "")))
                 + " · Normalization: " + escape(str(settings.get("normalization", "leaderboard"))) + "</p>")
    parts.append("<p>WER and CER pool edits over reference words and characters. Semantic distance averages "
                 "1 − cosine similarity across clips. Lower is better; semantic distance is not a percentage "
                 "or a calibrated measure of correctness. This report reuses the evaluation's scores.</p>")
    parts.append(table(["Model", "Checkpoint", "Prediction file"],
                       [[s["model"], s["model_id"], Path(s["file"]).name] for s in result["sources"]]))
    parts.append(table(["Model", "WER", "CER", "Semantic distance"],
                       [metric_row(row) for row in result["summary"]]))
    parts.append(f"<p class='note'>Clips per model: {result['summary'][0]['clips']}. These are supplied-data "
                 "scores. This page makes no population extrapolation or significance claim.</p>")
    if result["pairwise"]:
        parts.append("<h2>Where the metrics prefer different models</h2>")
        parts.append(table(["Pair", "Total clips", "Comparable", "Reversals"],
                           [[" / ".join(s["models"]), s["total"], s["comparable"], s["reversals"]]
                            for s in result["pairwise"]]))
        parts.append("<p>Comparable pairs have defined WER and no ties on either metric. A reversal means the "
                     "metrics prefer different predictions for the same clip. Pairs share clips and models; "
                     "they are not independent trials. The JSON retains all tie and exclusion counts.</p>")
    parts.append("<h2>Examples to review</h2><p>" + escape(settings.get("selection_rule", ""))
                 + " Examples are selected after scoring; they do not establish which metric is right.</p>")
    if not result["examples"]:
        parts.append("<p>No examples matched the selection rule.</p>")
    index = {(r["model"], r["id"]): r for r in result["records"]}
    for number, example in enumerate(result["examples"], 1):
        rows = [index[m, example["id"]] for m in example["models"]]
        parts.append("<details" + (" open" if number == 1 else "") + "><summary>Example "
                     + str(number) + " · " + escape(example["kind"]) + "</summary><div class='detail-body'>")
        parts.append("<p class='clip'>" + escape(example["id"]) + "</p><p><strong>Reference:</strong> "
                     + escape(rows[0]["reference"]) + "</p>")
        for row in rows:
            parts.append("<p><strong>" + escape(row["model"]) + ":</strong> "
                         + escape(row["prediction"] or "[empty]") + "</p>")
        parts.append(table(["Model", "WER", "CER", "Semantic distance"], [metric_row(r) for r in rows]))
        if "wer_prefers" in example:
            parts.append("<p>WER prefers " + escape(example["wer_prefers"]) + "; semantic distance prefers "
                         + escape(example["semantic_prefers"]) + ".</p>")
        else:
            parts.append(f"<p>Within this model, WER percentile: {100 * example['wer_percentile']:.1f}; "
                         f"semantic-distance percentile: {100 * example['semantic_percentile']:.1f}. "
                         "Higher percentiles mean worse relative scores, not probabilities.</p>")
        parts.append("<p class='note'>Check what information changed and whether it matters for the intended use. "
                     "Listen before attributing a mismatch to the model or reference.</p></div></details>")
    parts.append("<h2>Method and limits</h2><p>The English text encoder uses lightly cleaned original text, "
                 "independently of WER/CER normalization. CER includes single spaces. Identical semantic inputs "
                 "score 0; exactly one empty input scores 1 by convention. Empty normalized references have "
                 "undefined per-clip WER/CER; their insertions still contribute to corpus totals.</p>")
    parts.append("<p class='callout'>Numbers, names, and negation can be wrong despite a small semantic distance. "
                 "These text comparisons contain no independent audio judgment or human meaning labels. "
                 "Use the examples to direct review, not to certify correctness.</p>")
    parts.append("<details><summary>Settings and source provenance</summary><div class='detail-body'>")
    parts.append(table(["Setting", "Value"], [[k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                                             else v] for k, v in settings.items()]))
    parts.append(table(["Model", "Input SHA-256"], [[s["model"], s["sha256"]] for s in result["sources"]]))
    parts.append("</div></details><p class='note'>"
                 "<a href='https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2'>MiniLM model documentation</a> · "
                 "<a href='https://www.isca-archive.org/interspeech_2021/kim21e_interspeech.html'>SemDist paper</a> · "
                 "<a href='" + escape(path.with_suffix(".metrics.json").name, quote=True)
                 + "'>Full metrics and examples JSON</a></p></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
