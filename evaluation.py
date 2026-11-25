"""One evaluation pass: saved text -> metrics, figures, and HTML reports."""

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import numpy as np
from kaldialign import edit_distance

from analyze_transcripts import load_transcripts, build_analysis, write_report
from leaderboard_normalizer.normalizer import remove_non_speech_annotations
from normalization import LevelNormalizer, LEVEL_LABELS, NORMALIZATION_REVISION

NORMALIZER_REVISION = "c4696193a333f7948cb113dc93bb54f661bd9a0b"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def semantic_distances(pairs, batch_size=32):
    """Encode unique texts once on CPU; reuse distances for every policy."""
    import torch
    import transformers

    cleaned = [tuple(" ".join(remove_non_speech_annotations(t).split()) for t in pair)
               for pair in pairs]
    texts = list(dict.fromkeys(t for pair in cleaned for t in pair if t))
    vectors = {}
    torch.set_num_threads(2)
    if texts:
        print(f"Encoding {len(texts)} unique texts with MiniLM on CPU.", flush=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION)
        model = transformers.AutoModel.from_pretrained(
            ENCODER, revision=ENCODER_REVISION, use_safetensors=True,
            attn_implementation="eager",
        ).to(device="cpu", dtype=torch.float32).eval()
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] > 256:
                raise ValueError("A semantic input exceeds 256 tokens; nothing was truncated.")
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                if not torch.isfinite(pooled).all() or (pooled.norm(dim=1) == 0).any():
                    raise ValueError("Invalid semantic embeddings.")
                encoded = torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()
            vectors.update(zip(batch, encoded))
    distances = []
    for ref, hyp in cleaned:
        if ref == hyp:
            distances.append(0.0)
        elif not ref or not hyp:
            distances.append(1.0)  # Explicit convention for exactly one empty text.
        else:
            distances.append(float(1 - np.clip(np.dot(vectors[ref], vectors[hyp]), -1, 1)))
    settings = {
        "method": "1 - cosine similarity (SemDist-style, MiniLM encoder)",
        "encoder": ENCODER, "revision": ENCODER_REVISION,
        "pooling": "attention-mask mean, then L2 normalization",
        "preprocessing": "remove non-speech annotations; collapse whitespace",
        "independent_of_normalization_level": True,
        "max_tokens": 256, "truncation": False,
        "both_empty_distance": 0.0, "one_empty_distance": 1.0,
        "device": "cpu", "dtype": "float32", "batch_size": batch_size,
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
    }
    return distances, settings


def score_records(run, level, distances):
    """Compute word/character edits once; reports reuse these per-clip scores."""
    normalize = LevelNormalizer(level)
    scored = []
    for row, distance in zip(run["records"].values(), distances, strict=True):
        ref, hyp = [" ".join(normalize(row[k]).split()) for k in ("reference", "prediction")]
        words = edit_distance(ref.split(), hyp.split(), merge_compounds=normalize.merge_compounds)
        chars = edit_distance(list(ref), list(hyp), merge_compounds=False)
        words = {k: int(words[k]) for k in ("ins", "del", "sub", "total", "ref_len")}
        chars = {k: int(chars[k]) for k in ("ins", "del", "sub", "total", "ref_len")}
        scored.append({
            **row, "model": run["model"], "normalized_reference": ref, "normalized_prediction": hyp,
            "word_edits": words, "character_edits": chars,
            "wer_percent": 100 * words["total"] / words["ref_len"] if words["ref_len"] else None,
            "cer_percent": 100 * chars["total"] / chars["ref_len"] if chars["ref_len"] else None,
            "semantic_distance": distance,
        })
    return scored


def aggregate_scores(records, repeats=2000, seed=17):
    """Corpus scores and the original stratified finite-population bootstrap."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["meeting"], row["duration_group"])].append(row)
    rng = np.random.default_rng(seed)
    observed, estimated = np.zeros(2), np.zeros(2)
    bootstrap = np.zeros((repeats, 2))
    for group in sorted(groups):
        rows = groups[group]
        population, n = rows[0]["group_size"], len(rows)
        if any(r["group_size"] != population or r["sample_group_size"] != n for r in rows):
            raise ValueError(f"Incomplete or inconsistent group: {group}")
        if n > population or (n < 2 and n < population):
            raise ValueError(f"Insufficient observations for group: {group}")
        values = np.array([[r["word_edits"]["total"], r["word_edits"]["ref_len"]]
                           for r in rows], dtype=float)
        observed += values.sum(axis=0)
        mean = values.mean(axis=0)
        estimated += population * mean
        if n == population:
            bootstrap += population * mean
        else:
            draws = rng.integers(n, size=(repeats, n - 1))
            corrected = mean + np.sqrt(1 - n / population) * (values[draws].mean(axis=1) - mean)
            bootstrap += population * corrected
    if observed[1] <= 0 or np.any(bootstrap[:, 1] <= 0):
        raise ValueError("Too few normalized reference words for WER.")
    chars = {k: sum(r["character_edits"][k] for r in records)
             for k in ("total", "ref_len", "ins", "del", "sub")}
    if chars["ref_len"] <= 0:
        raise ValueError("Too few reference characters for CER.")
    distances = [r["semantic_distance"] for r in records]
    weights = [r["group_size"] / r["sample_group_size"] for r in records]
    return {
        "scored_clips": len(records), "sample_wer_percent": 100 * observed[0] / observed[1],
        "estimated_full_ami_wer_percent": 100 * estimated[0] / estimated[1],
        "full_ami_95_percent_interval": np.quantile(100 * bootstrap[:, 0] / bootstrap[:, 1], [.025, .975]).tolist(),
        "word_errors": int(observed[0]), "reference_words": int(observed[1]),
        **{name: sum(r["word_edits"][key] for r in records)
           for name, key in [("insertions", "ins"), ("deletions", "del"), ("substitutions", "sub")]},
        "bootstrap_repeats": repeats, "bootstrap_seed": seed,
        "sample_cer_percent": 100 * chars["total"] / chars["ref_len"],
        "character_errors": chars["total"], "reference_characters": chars["ref_len"],
        "character_insertions": chars["ins"], "character_deletions": chars["del"],
        "character_substitutions": chars["sub"], "cer_includes_spaces": True,
        "sample_semantic_distance": float(np.mean(distances)),
        "estimated_full_ami_semantic_distance": float(np.average(distances, weights=weights)),
        **{"semantic_empty_" + label: sum(not remove_non_speech_annotations(r[key]).strip() for r in records)
           for label, key in [("references", "reference"), ("predictions", "prediction")]},
    }


def plot_scores(comparison, output):
    """Save the same three figures, with consistent model colors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = [("sample_wer_percent", "WER (%)", "wer"),
             ("sample_cer_percent", "CER (%)", "cer"),
             ("sample_semantic_distance", "Mean semantic distance", "semantic_distance")]
    models = list(dict.fromkeys(r["predictions_file"] for r in comparison))
    for metric, label, name in plots:
        fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
        for i, model in enumerate(models):
            rows = [r for r in comparison if r["predictions_file"] == model]
            ax.plot([r["normalization_level"] for r in rows], [r[metric] for r in rows],
                    marker="o", color=f"C{i}", label=rows[0]["model_id"].rsplit("/", 1)[-1])
        ax.set_ylabel(label)
        ax.set_ylim(0, max(.01, max(r[metric] for r in comparison) * 1.15))
        ax.grid(axis="y", alpha=.25)
        ax.legend(loc="upper left", ncols=min(3, len(models)))
        ax.set_xticks(range(9), [f"{i}: {name}" for i, name in enumerate(LEVEL_LABELS)],
                      rotation=35, ha="right")
        ax.set_xlabel("Cumulative scoring strictness (forgiving → strict)")
        title = f"{label} on the same {comparison[0]['scored_clips']} clips — lower is better"
        if name == "semantic_distance":
            title += "\nOriginal text; independent of normalization"
        ax.set_title(title, fontsize=12)
        fig.savefig(output / f"{name}_by_normalization.png", dpi=160)
        plt.close(fig)


def write_overview(comparison, output):
    """A compact landing page for this evaluation, separate from the authored report."""
    baseline = [r for r in comparison if r["normalization_level"] == 0]
    rows = []
    for row in baseline:
        interval = "–".join(f"{v:.2f}" for v in row["full_ami_95_percent_interval"])
        cells = [row["model_id"], f"{row['sample_wer_percent']:.2f}%",
                 f"{row['sample_cer_percent']:.2f}%", f"{row['sample_semantic_distance']:.4f}",
                 f"{row['estimated_full_ami_wer_percent']:.2f}%", interval]
        rows.append("<tr>" + "".join("<td>" + escape(x) + "</td>" for x in cells) + "</tr>")
    figures = "".join(f'<h2>{label}</h2><img src="{name}_by_normalization.png" alt="{label}">'
                      for name, label in [("wer", "WER by normalization"), ("cer", "CER by normalization"),
                                          ("semantic_distance", "Semantic distance (original text)")])
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>ASR evaluation</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172c38}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ddd;text-align:left}img{width:100%}.scroll{overflow:auto}</style>'
    page += f'<h1>Saved-transcript evaluation</h1><p>{baseline[0]["scored_clips"]} shared clips. Lower scores are better.</p>'
    page += '<p><a href="transcript_analysis.html">Metric disagreements</a> · <a href="metadata_analysis.html">Metadata and listening review</a> · <a href="normalization.metrics.json">All scores</a></p>'
    page += '<div class="scroll"><table><tr><th>Model</th><th>Sample WER</th><th>CER</th><th>Semantic distance</th><th>Estimated corpus WER</th><th>Approx. 95% interval</th></tr>' + "".join(rows) + '</table></div>'
    page += '<p>Intervals describe sampling uncertainty for the weighted fixed-corpus WER estimate. They do not cover new meetings or reference errors. Semantic distance is not a percentage or a factual-accuracy guarantee. Its flat normalization curve follows from using original text.</p>'
    (output / "evaluation.html").write_text(page + figures + "</html>", encoding="utf-8")


def evaluate(files, output=Path("results"), repeats=2000, seed=17, text_batch_size=32):
    """Run all nine policies, share scores, then write all analysis reports."""
    from analyze_metadata import analyze

    if repeats < 1 or text_batch_size < 1:
        raise ValueError("Bootstrap repeats and text batch size must be positive.")
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError("The output directory is a file.")
    products = ["normalization.metrics.json", "transcript_analysis.metrics.json",
                "transcript_analysis.html", "metadata_analysis.metrics.json",
                "metadata_analysis.review.json", "metadata_analysis.html", "evaluation.html",
                "wer_by_normalization.png", "cer_by_normalization.png",
                "semantic_distance_by_normalization.png"]
    targets = {(output / name).resolve() for name in products}
    if any(Path(path).resolve() in targets for path in files):
        raise ValueError("Prediction inputs must not overlap generated output paths.")
    runs = load_transcripts(files)
    # Same IDs/references are checked by the loader; also compare sampling weights.
    fields = ("meeting", "duration_group", "group_size", "sample_group_size")
    samples = []
    for run in runs:
        try:
            samples.append({key: tuple(row[f] for f in fields) for key, row in run["records"].items()})
        except KeyError as error:
            raise ValueError(f"Missing sampling metadata: {error}. Use the prediction outputs.") from error
    if any(sample != samples[0] for sample in samples[1:]):
        raise ValueError("Models must use identical sampling groups and weights.")
    groups = defaultdict(list)
    for row in runs[0]["records"].values():
        if not all(isinstance(row[f], str) for f in ("meeting", "duration_group")):
            raise ValueError("Meeting and duration-group labels must be strings.")
        groups[row["meeting"], row["duration_group"]].append(row)
    for group, rows in groups.items():
        population, n = rows[0]["group_size"], len(rows)
        if (type(population) is not int or population < n
                or any(type(r["sample_group_size"]) is not int
                       or r["group_size"] != population or r["sample_group_size"] != n for r in rows)
                or (n < 2 and n < population)):
            raise ValueError(f"Incomplete or inconsistent sampling group: {group}")
    # Reject missing/inconsistent metadata before loading the text encoder.
    from analyze_metadata import REVISION
    if any(r["dataset_revision"] != REVISION for r in runs):
        raise ValueError("This evaluation uses the pinned AMI-Cleaned dataset only.")
    pairs = [(r["reference"], r["prediction"]) for run in runs for r in run["records"].values()]
    distances, semantic_settings = semantic_distances(pairs, text_batch_size)
    comparison, baseline_records, offset = [], [], 0
    for run in runs:
        n = len(run["records"])
        model_distances = distances[offset:offset+n]
        offset += n
        for level in range(9):
            scored = score_records(run, level, model_distances)
            row = aggregate_scores(scored, repeats, seed)
            row.update(normalization=f"level{level}", normalization_level=level,
                       normalization_revision=NORMALIZATION_REVISION,
                       normalizer_revision=NORMALIZER_REVISION, merge_compounds=level < 6,
                       model_id=run["model_id"], predictions_file=run["file"],
                       semantic_settings=semantic_settings)
            comparison.append(row)
            if level == 0:
                baseline_records.extend(scored)
            print(f"{run['model_id']} level{level}: WER {row['sample_wer_percent']:.2f}% "
                  f"CER {row['sample_cer_percent']:.2f}% distance {row['sample_semantic_distance']:.4f}", flush=True)
    # Review compatibility must pass before replacing any other analysis outputs.
    analyze(runs, baseline_records, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalization.metrics.json").write_text(json.dumps(comparison, indent=2) + "\n")
    plot_scores(comparison, output)
    settings = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "normalization": "level0", "normalization_revision": NORMALIZATION_REVISION,
                "normalizer_revision": NORMALIZER_REVISION, "merge_compounds": True,
                "cer_includes_spaces": True, **semantic_settings}
    analysis = build_analysis(runs, baseline_records, settings)
    write_report(analysis, output / "transcript_analysis.html")
    (output / "transcript_analysis.metrics.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_overview(comparison, output)
    print(f"Open {output / 'evaluation.html'}", flush=True)
    return comparison
