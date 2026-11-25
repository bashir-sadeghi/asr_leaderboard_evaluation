"""Select a fixed AMI sample and save resumable CPU/CUDA predictions."""

from collections import defaultdict
from dataclasses import replace
from importlib.metadata import version
from io import BytesIO
from itertools import islice
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq

from leaderboard_normalizer.normalizer import EnglishTextNormalizer

DATASET_ID = "hf-audio/open-asr-leaderboard"
DATASET_REVISION = "de79e57adf9983993bcc72586d7313460825f5eb"
NORMALIZER_REVISION = "c4696193a333f7948cb113dc93bb54f661bd9a0b"
DATA_FILES = [f"ami_cleaned/test-{i:05d}-of-00003.parquet" for i in range(3)]
MODELS = {
    "turbo": ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
    "large-v3": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "futo": ("futo-org/asr4all-l", "79d1c23bc8f1174cee576b40bbaa0118bd95abed"),
}


def sample_clips(sample_size=500, seed=17):
    """Select nested samples proportionally across meeting-duration groups."""
    clips = []
    normalize = EnglishTextNormalizer()
    for filename in DATA_FILES:
        path = hf_hub_download(
            DATASET_ID, filename, repo_type="dataset", revision=DATASET_REVISION
        )
        clips.extend(pq.read_table(path, columns=["id", "text"]).to_pylist())

    eligible = []
    for clip in clips:
        normalized_text = normalize(clip["text"]).strip()
        if normalized_text and normalized_text != "ignore time segment in scoring":
            eligible.append(clip)
    print(f"Eligible clips {len(eligible)} of {len(clips)}", flush=True)
    clips = eligible
    if not 1 <= sample_size <= len(clips):
        raise ValueError(f"Sample size must be between 1 and {len(clips)}.")

    for clip in clips:
        parts = clip["id"].split("_")
        clip["meeting"] = parts[1]
        clip["duration"] = (int(parts[-1]) - int(parts[-2])) / 100
    cutoffs = np.quantile([clip["duration"] for clip in clips], [1/3, 2/3])

    groups = defaultdict(list)
    for clip in clips:
        if clip["duration"] <= cutoffs[0]:
            clip["duration_group"] = "short"
        elif clip["duration"] <= cutoffs[1]:
            clip["duration_group"] = "medium"
        else:
            clip["duration_group"] = "long"
        groups[(clip["meeting"], clip["duration_group"])].append(clip)

    keys = sorted(groups)
    sizes = np.array([len(groups[key]) for key in keys])
    counts = np.minimum(2, sizes)
    if sample_size < counts.sum():
        raise ValueError("Increase sample size so each group has at least two clips for bootstrap.")
    # Add one clip at a time to the group furthest below its proportional quota.
    # Counts only increase, so extending the sample never removes a clip.
    for total in range(int(counts.sum()), sample_size):
        deficits = (total + 1) * sizes / sizes.sum() - counts
        deficits[counts == sizes] = -np.inf
        counts[np.argmax(deficits)] += 1

    rng = np.random.default_rng(seed)
    selected_ids = set()
    for key, count in zip(keys, counts):
        group = groups[key]
        # Shuffle the entire group once; every sample size uses the same order.
        for index in rng.permutation(len(group))[:count]:
            clip = group[index]
            clip["group_size"] = len(group)
            clip["sample_group_size"] = int(count)
            selected_ids.add(clip["id"])

    # Keep dataset order, which is sorted by duration and helps batching.
    selected = [clip for clip in clips if clip["id"] in selected_ids]
    print(f"Selected {len(selected)} clips from {len(groups)} groups. Cutoffs {cutoffs}.", flush=True)
    return selected

def resolve_device(device="auto"):
    """Validate the requested hardware before downloading data or loading models."""
    import torch

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select --device cpu or enable a GPU runtime.")
    return device


def predict(clips, model_name, batch_size, output, device="auto", seed=17):
    """Resume or extend this output file; never reuse a different device's run."""
    import torch

    if not clips or batch_size < 1:
        raise ValueError("Provide clips and a positive batch size.")
    device = resolve_device(device)
    dtype = torch.float32 if device == "cpu" or model_name == "futo" else torch.float16
    output = Path(output)
    model_id, revision = MODELS[model_name]
    expected = {clip["id"]: clip for clip in clips}
    if len(expected) != len(clips):
        raise ValueError("Selected clip IDs must be unique.")

    if model_name == "futo":
        # The leaderboard uses a different revision and tier (256, 64).
        decoding = {
            "punctuate": True, "op": [256, 32], "attention": "manual",
            "feature_padding": 0.0, "implementation": "batched_features_cpu_pcec_v1",
        }
    else:
        decoding = {
            "language": "en", "task": "transcribe", "num_beams": 1,
            "do_sample": False, "max_new_tokens": 444,
        }
    settings = {
        "dataset_revision": DATASET_REVISION, "normalizer_revision": NORMALIZER_REVISION,
        "model_id": model_id, "model_revision": revision,
        "seed": seed, "sampling": "nested_v1", "decoding": decoding,
        "inference": {"device": device, "batch_size": batch_size, "dtype": str(dtype)},
        "sample_ids": list(expected),
    }
    document = {"settings": settings, "runs": [], "predictions": []}
    if output.exists():
        document = json.loads(output.read_text())
        previous = document["settings"]
        if {k: v for k, v in previous.items() if k != "sample_ids"} != {
            k: v for k, v in settings.items() if k != "sample_ids"
        }:
            raise ValueError("This output uses different settings. Choose another --output.")
        old_ids = previous["sample_ids"]
        if len(old_ids) != len(set(old_ids)) or not set(old_ids) <= expected.keys():
            raise ValueError("Resume with the same sample or a larger nested sample.")
        # Same IDs must retain the dataset order used for batching.
        if [key for key in expected if key in set(old_ids)] != old_ids:
            raise ValueError("The existing sample order differs.")
    completed = {}
    old_ids = set(document["settings"]["sample_ids"])
    for row in document["predictions"]:
        key = row["id"]
        if key in completed or key not in old_ids:
            raise ValueError("Saved predictions contain duplicate or unexpected IDs.")
        if not isinstance(row.get("prediction"), str):
            raise ValueError(f"Saved prediction is not text: {key}")
        if any(row.get(k) != v for k, v in expected[key].items() if k != "sample_group_size"):
            raise ValueError(f"Saved metadata differs for {key}.")
        # A larger sample changes group weights, not the existing transcript.
        completed[key] = {**row, **expected[key]}
    document["settings"] = settings

    def save():
        document["predictions"] = [completed[c["id"]] for c in clips if c["id"] in completed]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(output)

    pending = set(expected) - completed.keys()
    save()
    if not pending:
        print(f"Reused all {len(completed)} predictions from {output}.", flush=True)
        return document

    # Load the model only when clips still need transcription.
    import soundfile as sf
    if model_name == "futo":
        from transformers import AutoModel

        transcribe = AutoModel.from_pretrained(
            model_id, revision=revision, code_revision=revision,
            trust_remote_code=True, use_safetensors=True, dtype=dtype,
        ).eval()
        transcribe.encoder.to(device)
        transcribe.ctc_head.to(device)
        transcribe.encoder.force_op = tuple(decoding["op"])
        # Avoid the optional FlexAttention kernels; use the same path on CPU/CUDA.
        transcribe.encoder.ec = replace(transcribe.encoder.ec, use_flex_attn=False)
        for module in transcribe.encoder.modules():
            if hasattr(module, "manual_attention"):
                module.manual_attention = True
        if transcribe.pcec is None:
            raise ValueError("The FUTO checkpoint must include its correction head.")
    else:
        from transformers import pipeline

        transcribe = pipeline(
            "automatic-speech-recognition", model=model_id, revision=revision,
            device=device, dtype=dtype,
        )

    run = {
        "device": device, "batch_size": batch_size, "dtype": str(dtype),
        "torch": torch.__version__, "transformers": version("transformers"),
    }
    if model_name == "futo":
        run.update(frontend_device="cpu", correction_device="cpu", torchaudio=version("torchaudio"))
    document["runs"].append(run)
    print(f"Running {model_name} on {device}, batch size {batch_size}; "
          f"{len(completed)} reused, {len(pending)} remaining.", flush=True)

    for filename in DATA_FILES:
        path = hf_hub_download(
            DATASET_ID, filename, repo_type="dataset", revision=DATASET_REVISION
        )
        with pq.ParquetFile(path) as shard:
            rows = (
                row for table in shard.iter_batches(batch_size=32, columns=["id", "audio"])
                for row in table.to_pylist() if row["id"] in pending
            )
            while batch := list(islice(rows, batch_size)):
                waveforms = []
                for row in batch:
                    audio, sample_rate = sf.read(
                        BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True
                    )
                    if sample_rate != 16000:
                        raise ValueError(f"Expected 16 kHz AMI audio for {row['id']}.")
                    waveforms.append(audio.mean(axis=1))

                if model_name == "futo":
                    with torch.inference_mode():
                        # Features and correction stay on CPU; only encoder/CTC use CUDA.
                        features = [
                            transcribe.pre(torch.from_numpy(audio).unsqueeze(0))[0].squeeze(0)
                            for audio in waveforms
                        ]
                        lengths = torch.tensor([len(feature) for feature in features])
                        padded = torch.nn.utils.rnn.pad_sequence(
                            features, batch_first=True, padding_value=decoding["feature_padding"]
                        ).to(device)
                        encoded = transcribe.encoder(padded, lengths)
                        logits = transcribe.ctc_head(encoded.hidden).cpu()
                        hidden = encoded.hidden.cpu()
                        texts = [
                            transcribe._pcec_decode(logits[i:i+1, :n], hidden[i:i+1, :n])
                            for i, n in enumerate(encoded.lens.tolist())
                        ]
                        # This encoder has no padding-key mask: batch mates can affect text.
                        batch_ids = [row["id"] for row in batch]
                        padded_frames = padded.shape[1]
                        del encoded, padded, logits, hidden
                else:
                    inputs = [{"raw": audio, "sampling_rate": 16000} for audio in waveforms]
                    texts = [r["text"] for r in transcribe(
                        inputs, batch_size=batch_size, generate_kwargs=decoding
                    )]
                for row, text in zip(batch, texts, strict=True):
                    key = row["id"]
                    completed[key] = {**expected[key], "prediction": text}
                    if model_name == "futo":
                        completed[key].update(
                            inference_batch_ids=batch_ids, padded_feature_frames=padded_frames
                        )
                    pending.remove(key)
                save()
                print(f"Saved {len(completed)}/{len(clips)} predictions.", flush=True)
    if pending:
        raise ValueError(f"Audio was missing for {len(pending)} clips.")
    return document
