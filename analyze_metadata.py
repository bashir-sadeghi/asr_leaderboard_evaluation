"""Analyze AMI metadata and saved predictions; read cached audio without running models."""

import base64
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
import pyarrow.parquet as pq
import soundfile as sf

from leaderboard_normalizer.normalizer import remove_non_speech_annotations
from normalization import LevelNormalizer

DATASET = "hf-audio/open-asr-leaderboard"
REVISION = "de79e57adf9983993bcc72586d7313460825f5eb"
# Independent catalog, transcribed from the pinned CSTR builder's _EVAL_SAMPLE_IDS.
MEETINGS = {family + suffix for family in ("EN2002", "ES2004", "IS1009", "TS3003") for suffix in "abcd"}
CATALOG = "https://huggingface.co/datasets/edinburghcstr/ami/blob/fc3b6e8c8cbdcfa11d476e07aa338d937586bd1c/ami-ihm-kaldi-chunked.py"
AMI_SPLIT = "https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml"
NORMALIZER_REVISION = "c4696193a333f7948cb113dc93bb54f661bd9a0b"
ID_PATTERN = re.compile(r"AMI_([A-Z]{2}\d{4}[a-z]?)_H\d+_[^_]+_(\d+)_(\d+)")


def read_metadata(references, normalize, tolerance, download):
    """Measure decoded frames, not total samples across channels. Keep compressed audio for review."""
    metadata, audio_bytes, shards = {}, {}, []
    for i in range(3):
        filename = f"ami_cleaned/test-{i:05d}-of-00003.parquet"
        try:
            path = hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION,
                                   local_files_only=not download)
        except LocalEntryNotFoundError as error:
            raise ValueError("AMI audio is missing from the cache.") from error
        shards.append(filename)
        print(f"Reading {filename}", flush=True)
        with pq.ParquetFile(path) as shard:
            for batch in shard.iter_batches(batch_size=16, columns=["id", "text", "audio", "audio_length_s"]):
                for row in batch.to_pylist():
                    clip_id = row["id"]
                    if clip_id not in references:
                        continue
                    if clip_id in metadata or row["text"] != references[clip_id]:
                        raise ValueError(f"Duplicate clip or source-reference mismatch: {clip_id}")
                    match = ID_PATTERN.fullmatch(clip_id)
                    if not match or int(match[3]) <= int(match[2]):
                        raise ValueError(f"Unexpected AMI ID: {clip_id}")
                    content = row["audio"]["bytes"]
                    waveform, sample_rate = sf.read(BytesIO(content), dtype="float32", always_2d=True)
                    if not len(waveform) or sample_rate <= 0 or not np.isfinite(waveform).all():
                        raise ValueError(f"Invalid decoded audio: {clip_id}")
                    duration = len(waveform) / sample_rate
                    id_duration = (int(match[3]) - int(match[2])) / 100
                    source_duration = row["audio_length_s"]
                    if source_duration is None or not np.isfinite(source_duration) or source_duration <= 0:
                        raise ValueError(f"Invalid source audio_length_s: {clip_id}")
                    raw_words = len(remove_non_speech_annotations(row["text"]).split())
                    normalized_words = len(normalize(row["text"]).split())
                    metadata[clip_id] = {
                        "id": clip_id, "reference": row["text"], "meeting": match[1],
                        "meeting_in_catalog": match[1] in MEETINGS,
                        "audio_seconds": duration, "frames": len(waveform), "sample_rate": sample_rate,
                        "id_seconds": id_duration, "source_seconds": float(source_duration),
                        "id_duration_error_seconds": abs(duration - id_duration),
                        "source_duration_error_seconds": abs(duration - source_duration),
                        "id_duration_ok": abs(duration - id_duration) <= tolerance + 1e-9,
                        "source_duration_ok": abs(duration - source_duration) <= tolerance + 1e-9,
                        "raw_words": raw_words, "normalized_words": normalized_words,
                        "raw_density": raw_words / duration, "normalized_density": normalized_words / duration,
                        "audio_sha256": sha256(content).hexdigest(), "shard": filename,
                    }
                    audio_bytes[clip_id] = content
        print(f"Measured {len(metadata)}/{len(references)} selected clips.", flush=True)
    if metadata.keys() != references.keys():
        raise ValueError(f"Audio missing for {len(references.keys() - metadata.keys())} clips; none were dropped.")
    return list(metadata.values()), audio_bytes, shards


def make_groups(metadata):
    """Common sample-based thirds; tied values stay together, so sizes may differ."""
    cutoffs = {}
    for field, group, labels in [
        ("audio_seconds", "duration_group", ["short", "medium", "long"]),
        ("normalized_density", "density_group", ["low", "medium", "high"]),
        ("raw_density", "raw_density_group", ["low", "medium", "high"]),
    ]:
        cuts = np.quantile([r[field] for r in metadata], [1/3, 2/3])
        cutoffs[group] = cuts.tolist()
        for row in metadata:
            row[group] = labels[int(np.searchsorted(cuts, row[field], side="left"))]
    for row in metadata:
        row["id_duration_group"] = ["short", "medium", "long"][int(np.searchsorted(
            cutoffs["duration_group"], row["id_seconds"], side="left"))]
    return cutoffs


def summarize(runs, metadata, scored_records, tolerance):
    """Keep WER normalization fixed while changing only density group assignments."""
    by_id = {r["id"]: r for r in metadata}
    groups = {"overall": ["all"], "duration_group": ["short", "medium", "long"],
              "density_group": ["low", "medium", "high"], "raw_density_group": ["low", "medium", "high"],
              "meeting": sorted({r["meeting"] for r in metadata})}
    records, tables, saved_checks = [], [], []
    scores = {(r["model"], r["id"]): r["word_edits"] for r in scored_records}
    for run in runs:
        saved = json.loads(Path(run["file"]).read_text())
        original = saved["predictions"] if isinstance(saved, dict) else saved
        mismatches = []
        for row in original:
            meta = by_id[row["id"]]
            issues = []
            if "meeting" in row and row["meeting"] != meta["meeting"]:
                issues.append("saved meeting differs from ID")
            if "duration" in row:
                value = row["duration"]
                if (not isinstance(value, (int, float)) or not np.isfinite(value)
                        or abs(value - meta["audio_seconds"]) > tolerance + 1e-9):
                    issues.append("saved duration differs from decoded audio")
            if issues: mismatches.append({"id": row["id"], "issues": issues})
        saved_checks.append({"model": run["model"], "mismatches": mismatches,
                             "saved_meeting_present": sum("meeting" in r for r in original),
                             "saved_duration_present": sum("duration" in r for r in original)})
        for row in run["records"].values():
            counts = scores[run["model"], row["id"]]
            records.append({"id": row["id"], "model": run["model"],
                            **{k: int(counts[k]) for k in ("ins", "del", "sub", "total", "ref_len")}})
        model_rows = [r for r in records if r["model"] == run["model"]]
        for field, labels in groups.items():
            for label in labels:
                rows = [r for r in model_rows if field == "overall" or by_id[r["id"]][field] == label]
                errors, words = sum(r["total"] for r in rows), sum(r["ref_len"] for r in rows)
                tables.append({"model": run["model"], "grouping": field, "group": label,
                               "clips": len(rows), "reference_words": words, "word_errors": errors,
                               "wer_percent": 100 * errors / words if words else None,
                               "mean_audio_seconds": float(np.mean([by_id[r["id"]]["audio_seconds"] for r in rows])) if rows else None})
    return tables, records, saved_checks


def prepare_review(metadata, path, limit):
    """Select extremes without looking at predictions; preserve any human answers."""
    orders = [sorted(metadata, key=lambda r: (r["audio_seconds"], r["id"])),
              sorted(metadata, key=lambda r: (-r["audio_seconds"], r["id"])),
              sorted(metadata, key=lambda r: (r["normalized_density"], r["id"])),
              sorted(metadata, key=lambda r: (-r["normalized_density"], r["id"])),
              sorted(metadata, key=lambda r: (-abs(r["raw_density"] - r["normalized_density"]), r["id"]))]
    reasons = ["short duration", "long duration", "low word density", "high word density", "normalization-sensitive density"]
    chosen = {}
    for candidates in zip(*orders):
        for row, reason in zip(candidates, reasons):
            if len(chosen) < min(limit, len(metadata)) and row["id"] not in chosen:
                chosen[row["id"]] = {"id": row["id"], "reference": row["reference"], "reason": reason,
                                     "reference_matches_audio": None, "corrected_reference": None, "notes": ""}
        if len(chosen) == min(limit, len(metadata)): break
    if path.exists():
        review = json.loads(path.read_text())
        items = review.get("clips", [])
        if review.get("dataset_revision") != REVISION or len(items) != len(chosen) or {r["id"] for r in items} != set(chosen):
            raise ValueError("Existing review belongs to a different selection. Preserve it and choose a new --output.")
        for row in items:
            if row["reference"] != chosen[row["id"]]["reference"]:
                raise ValueError(f"Do not edit the original review reference: {row['id']}")
            answer = row.get("reference_matches_audio")
            if answer is not None and not isinstance(answer, bool):
                raise ValueError("reference_matches_audio must be true, false, or null.")
            if row.get("corrected_reference") is not None and not isinstance(row["corrected_reference"], str):
                raise ValueError("corrected_reference must be text or null.")
        return review
    return {"dataset_revision": REVISION,
            "instructions": "Listen using the HTML report. Set reference_matches_audio to true/false; null means pending. "
                            "Optionally fill corrected_reference and notes. Keep id and reference unchanged, then rerun the same command.",
            "selection": "Deterministic extremes of duration, density and normalization sensitivity, without model scores; not a random audit.",
            "clips": list(chosen.values())}


def write_html(result, audio_bytes, path, review_path):
    """A local report with playable WAV clips; no JavaScript or external services."""
    def table(headers, rows):
        return '<table><thead><tr>' + ''.join(f'<th>{escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
            '<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'

    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>AMI metadata analysis</title>',
             '<style>body{font:16px/1.5 system-ui;margin:40px auto;max-width:1100px;padding:0 24px;color:#18212b}'
             'table{border-collapse:collapse;width:100%;margin:16px 0 28px}th,td{border:1px solid #ddd;padding:8px;text-align:right}'
             'th:first-child,td:first-child{text-align:left}th{background:#e7eff5}article{border-top:1px solid #ddd;padding:18px 0}'
             'code{overflow-wrap:anywhere}audio{width:100%}.note{color:#475569}</style><body>',
             '<h1>AMI metadata analysis</h1>',
             '<p>Duration is decoded audio frames divided by sample rate. Word density is reference tokens per second, '
             'including pauses. Meeting labels come from clip IDs and are checked against an independent evaluation-session catalog.</p>',
             '<p>No ASR or text encoder runs. Group results describe these supplied clips; no confidence intervals or full-dataset estimates are implied.</p>']
    parts.append('<p>' + escape(result['settings'].get('measurement_note', '')) + '</p>')
    parts.append('<h2>Inputs</h2>')
    parts.append(table(['Model', 'Checkpoint label', 'Prediction file'], [[s['model'], s['model_id'], s['file']] for s in result['sources']]))
    parts.append('<h2>Automatic reliability checks</h2>')
    rel = result['reliability']; count = len(result['metadata'])
    parts.append(table(['Duration source', 'Within tolerance', 'Median absolute difference', 'Maximum absolute difference'],
                       [[s['source'], f"{s['within_tolerance']}/{count}", f"{s['median_abs_error_seconds']:.6f} s", f"{s['max_abs_error_seconds']:.6f} s"]
                        for s in rel['duration_checks']]))
    parts.append(f"<p>Tolerance: {result['settings']['duration_tolerance_seconds']:g} seconds. The default 0.02 s is a chosen allowance "
                 "for centisecond ID boundaries, not a proved annotation-error bound. This checks stored duration consistency, not whether speech was cut at the correct boundary.</p>")
    parts.append(f"<p>References matched the pinned dataset for all {count} clips. "
                 f"Meeting catalog membership: {count - rel['unknown_meeting_clips']}/{count}. "
                 "Catalog membership validates the session name, not the audio-to-meeting assignment.</p>")
    parts.append(table(['Model', 'Saved meeting labels', 'Saved durations', 'Clips with metadata mismatches'],
                       [[r['model'], r['saved_meeting_present'], r['saved_duration_present'], len(r['mismatches'])] for r in result['saved_metadata_checks']]))
    parts.append(f"<p>Applying the measured-duration cutoffs to ID-derived durations changes {rel['duration_group_changes']} clip groups. "
                 f"Counting words before versus after normalization changes {rel['density_group_changes']}/{count} density groups. "
                 "Density thirds are recomputed for each word-count definition; WER normalization remains fixed.</p>")
    parts.append(table(['Metadata grouping', 'First cutoff', 'Second cutoff'],
                       [[name, f'{cuts[0]:.4f}', f'{cuts[1]:.4f}'] for name, cuts in result['cutoffs'].items()]))
    parts.append('<p class="note">Cutoffs are computed on this shared sample, not the full AMI population. '
                 'Duration uses seconds; density uses words/second. Ties stay together, so groups can differ in size.</p>')
    parts.append(f"<p>Reference token totals: {rel['raw_reference_tokens']} before normalization and "
                 f"{rel['normalized_reference_tokens']} after normalization. "
                 f"Token counts change in {rel['word_count_changes']}/{count} clips.</p>")
    parts.append(table(['Normalized density / raw density', 'low', 'medium', 'high'],
                       [[label, *rel['density_group_transition'][label]] for label in ('low', 'medium', 'high')]))
    for key, title in [('overall', 'Overall WER'), ('duration_group', 'WER by measured duration'),
                       ('density_group', 'WER by normalized word density'), ('raw_density_group', 'Sensitivity using raw word density'),
                       ('meeting', 'WER by meeting')]:
        parts.append(f'<h2>{title}</h2>')
        parts.append(table(['Group', 'Model', 'Clips', 'Reference words', 'Word errors', 'WER'],
                           [[r['group'], r['model'], r['clips'], r['reference_words'], r['word_errors'],
                             'N/A' if r['wer_percent'] is None else f"{r['wer_percent']:.2f}%"] for r in result['breakdowns'] if r['grouping'] == key]))
    parts.append('<h2>Manual reference check</h2>')
    review = result['manual_review']; reviewed = sum(r['reference_matches_audio'] is not None for r in review['clips'])
    rejected = sum(r['reference_matches_audio'] is False for r in review['clips'])
    parts.append(f'<p>{reviewed}/{len(review["clips"])} clips reviewed: {reviewed - rejected} matching references, '
                 f'{rejected} mismatches, {len(review["clips"]) - reviewed} pending. '
                 'These human review records cover only the selected cases, not corpus-wide reference accuracy.</p>')
    parts.append('<p>Listen and compare with the original reference. Edit <code>' + escape(str(review_path)) +
                 '</code>: set <code>reference_matches_audio</code> to true or false, optionally add '
                 '<code>corrected_reference</code> and notes, then rerun the same command. Existing answers are preserved.</p>')
    by_id = {r['id']: r for r in result['metadata']}
    normalize = LevelNormalizer(0)
    for item in review['clips']:
        row = by_id[item['id']]
        waveform, sr = sf.read(BytesIO(audio_bytes[item['id']]), dtype='float32', always_2d=True)
        wav = BytesIO(); sf.write(wav, waveform, sr, format='WAV', subtype='PCM_16')
        encoded = base64.b64encode(wav.getvalue()).decode('ascii')
        parts.append('<article><h3>' + escape(item['id']) + '</h3><p>' + escape(item['reason']) +
                     f" · {row['audio_seconds']:.2f} s · {row['normalized_density']:.2f} normalized words/s</p>")
        parts.append('<p><strong>Reference:</strong> ' + escape(item['reference']) + '</p>')
        parts.append(f'<audio controls preload="none" src="data:audio/wav;base64,{encoded}"></audio>')
        status = 'pending' if item['reference_matches_audio'] is None else str(item['reference_matches_audio']).lower()
        parts.append('<p>Review status: ' + status + '. Notes: ' + escape(str(item.get('notes', ''))) + '</p>')
        if item.get('corrected_reference') is not None:
            corrected_words = len(normalize(item['corrected_reference']).split())
            parts.append('<p>Reviewer correction: ' + escape(item['corrected_reference']) +
                         f"</p><p>Normalized word count: {row['normalized_words']} → {corrected_words}. "
                         f"Density: {row['normalized_density']:.2f} → {corrected_words / row['audio_seconds']:.2f} words/s. "
                         "The main scores and group assignments still use the original reference.</p>")
        parts.append('</article>')
    parts.append('<h2>Interpretation and limits</h2><p>Word density includes pauses and depends on tokenization, fillers, '
                 'number rewriting and contraction expansion. It is not pure speaking speed. Word density and WER share '
                 'reference-word counts, so their association can partly reflect their formulas. Meetings can differ in speakers, '
                 'content and recording conditions; a difference in WER does not identify a cause. Small groups require caution.</p>')
    parts.append('<p>Duration differences, unknown meetings and saved-metadata mismatch details are retained in the companion JSON; '
                 'no flagged clips are silently removed. This report checks consistency and sensitivity, not independent acoustic truth.</p>')
    parts.append(f'<p>Sources: <a href="{CATALOG}">Pinned CSTR evaluation-session list</a>; '
                 f'<a href="{AMI_SPLIT}">Official AMI ASR split</a>; '
                 f'<a href="https://huggingface.co/datasets/{DATASET}/tree/{REVISION}/ami_cleaned">Pinned evaluation data</a>.</p></body></html>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def cached_measurements(references, folder):
    """Reuse measured audio metadata and the 12 embedded clips for this same sample."""
    path, html = folder / "metadata_analysis.metrics.json", folder / "metadata_analysis.html"
    if not path.exists() or not html.exists():
        return None
    saved = json.loads(path.read_text())
    settings = saved["settings"]
    rows = saved["metadata"]
    if (settings["dataset_revision"] != REVISION
            or settings["normalizer_revision"] != NORMALIZER_REVISION
            or settings["duration_tolerance_seconds"] != .02
            or len(rows) != len(references)
            or {r["id"]: r["reference"] for r in rows} != references):
        return None
    audio = {}
    for clip_id, encoded in re.findall(
            r'<article><h3>([^<]+)</h3>.*?src="data:audio/wav;base64,([^"]+)"', html.read_text(), re.S):
        audio[clip_id] = base64.b64decode(encoded, validate=True)
    if not {r["id"] for r in saved["manual_review"]["clips"]} <= audio.keys():
        return None
    return saved, audio


def analyze(runs, scored_records, output):
    """Write metadata results using baseline scores; preserve human listening answers."""
    if any(r["dataset_revision"] != REVISION for r in runs):
        raise ValueError("Metadata analysis supports the pinned AMI-Cleaned dataset.")
    metrics_path = output / "metadata_analysis.metrics.json"
    review_path = output / "metadata_analysis.review.json"
    references = {key: row["reference"] for key, row in runs[0]["records"].items()}
    normalize = LevelNormalizer(0)
    # Packaged measurements make re-scoring fast and avoid downloading audio again.
    cached = cached_measurements(references, output)
    if cached is None and output.resolve() != (Path(__file__).parent / "results").resolve():
        cached = cached_measurements(references, Path(__file__).parent / "results")
    if cached:
        saved, audio_bytes = cached
        metadata, shards = saved["metadata"], saved["settings"]["shards"]
        measured_at = saved["settings"].get("measurement_created_utc", saved["settings"]["created_utc"])
        note = "Reused saved duration measurements and review audio for the identical IDs/references."
        print(note, flush=True)
    else:
        print("Measuring this sample's audio; missing pinned shards will be downloaded.", flush=True)
        metadata, audio_bytes, shards = read_metadata(references, normalize, .02, True)
        measured_at = datetime.now(timezone.utc).isoformat()
        note = "Audio durations were measured during this evaluation."
    # Word counts are cheap to refresh even when audio measurements are reused.
    for row in metadata:
        row["raw_words"] = len(remove_non_speech_annotations(row["reference"]).split())
        row["normalized_words"] = len(normalize(row["reference"]).split())
        row["raw_density"] = row["raw_words"] / row["audio_seconds"]
        row["normalized_density"] = row["normalized_words"] / row["audio_seconds"]
    cutoffs = make_groups(metadata)
    review = prepare_review(metadata, review_path, 12)
    if cached and not review_path.exists():
        prior = saved["manual_review"]
        if {r["id"]: r["reference"] for r in prior["clips"]} == {
                r["id"]: r["reference"] for r in review["clips"]}:
            review = prior
    if not {r["id"] for r in review["clips"]} <= audio_bytes.keys():
        # Needed only if a changed selection rule chooses different review clips.
        _, audio_bytes, shards = read_metadata(references, normalize, .02, True)
    breakdowns, edits, saved_checks = summarize(runs, metadata, scored_records, .02)
    duration_checks = []
    for source in ("id", "source"):
        values = [r[f"{source}_duration_error_seconds"] for r in metadata]
        duration_checks.append({"source": source,
            "within_tolerance": sum(r[f"{source}_duration_ok"] for r in metadata),
            "median_abs_error_seconds": float(np.median(values)), "max_abs_error_seconds": max(values)})
    labels = ["low", "medium", "high"]
    transitions = {label: [sum(r["density_group"] == label and r["raw_density_group"] == other
                              for r in metadata) for other in labels] for label in labels}
    result = {
        "settings": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_created_utc": measured_at, "measurement_note": note,
            "dataset": DATASET, "dataset_revision": REVISION,
            "normalization": "leaderboard", "normalizer_revision": NORMALIZER_REVISION,
            "merge_compounds": True, "duration_tolerance_seconds": .02,
            "binning": "shared sample quantile thirds; ties stay together",
            "normalized_density": "leaderboard-normalized reference tokens / decoded audio seconds",
            "raw_density": "annotation-cleaned whitespace reference tokens / decoded audio seconds",
            "meeting_catalog": CATALOG, "expected_meetings": sorted(MEETINGS), "shards": shards,
            "manual_review_file": str(review_path.resolve()), "asr_inference": False,
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {p: version(p) for p in ("numpy", "soundfile", "pyarrow", "kaldialign")},
        },
        "sources": [{k: v for k, v in run.items() if k != "records"} for run in runs],
        "cutoffs": cutoffs, "metadata": metadata, "breakdowns": breakdowns, "word_edits": edits,
        "saved_metadata_checks": saved_checks,
        "reliability": {
            "duration_checks": duration_checks,
            "unknown_meeting_ids": sorted({r["meeting"] for r in metadata if not r["meeting_in_catalog"]}),
            "unknown_meeting_clips": sum(not r["meeting_in_catalog"] for r in metadata),
            "duration_group_changes": sum(r["duration_group"] != r["id_duration_group"] for r in metadata),
            "density_group_changes": sum(r["density_group"] != r["raw_density_group"] for r in metadata),
            "density_group_transition": transitions,
            "raw_reference_tokens": sum(r["raw_words"] for r in metadata),
            "normalized_reference_tokens": sum(r["normalized_words"] for r in metadata),
            "word_count_changes": sum(r["raw_words"] != r["normalized_words"] for r in metadata),
        },
        "manual_review": review,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_html(result, audio_bytes, output / "metadata_analysis.html", review_path)
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if not review_path.exists():
        review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n")
    return result
