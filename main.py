"""Two commands: predict audio once, then evaluate the saved transcripts."""

from argparse import ArgumentParser
from pathlib import Path

MODELS = ("turbo", "large-v3", "futo")
FRESH = Path("runs/fresh")
SUBMITTED = Path("results")


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prediction = commands.add_parser("predict", help="Transcribe and save/resume one model.")
    prediction.add_argument("--model", choices=MODELS, default="turbo")
    prediction.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    prediction.add_argument("--batch-size", type=int, default=4)
    prediction.add_argument("--sample-size", type=int, default=500)
    prediction.add_argument("--seed", type=int, default=17)
    prediction.add_argument("--output", type=Path, help="Default: runs/fresh/MODEL.json; also used to resume.")

    evaluation = commands.add_parser("evaluate", help="All metrics, plots, and HTML reports; no ASR.")
    evaluation.add_argument("files", type=Path, nargs="*", help="Optional saved prediction files.")
    evaluation.add_argument("--submitted", action="store_true", help="Score the included transcripts.")
    evaluation.add_argument("--output-dir", type=Path, help="Override the analysis output folder.")
    evaluation.add_argument("--bootstrap-repeats", type=int, default=2000)
    evaluation.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.command == "predict":
        from prediction import sample_clips, predict, resolve_device

        if args.batch_size < 1 or args.sample_size < 1:
            parser.error("Batch size and sample size must be positive.")
        device = resolve_device(args.device)
        clips = sample_clips(args.sample_size, args.seed)
        output = args.output or FRESH / f"{args.model}.json"
        predict(clips, args.model, args.batch_size, output, device, args.seed)
        print(f"Saved predictions: {output}")
        print(f"Evaluate this model: python main.py evaluate {output}")
    else:
        from evaluation import evaluate

        if args.files and args.submitted:
            parser.error("Choose explicit files or --submitted, not both.")
        if args.files:
            files = args.files
            output_dir = args.output_dir or Path("runs/recheck")
        elif args.submitted:
            files = [SUBMITTED / f"{model}.json" for model in MODELS]
            output_dir = args.output_dir or Path("runs/recheck")
        else:
            fresh = [FRESH / f"{model}.json" for model in MODELS]
            present = [path for path in fresh if path.exists()]
            if present and len(present) != len(fresh):
                missing = [str(path) for path in fresh if not path.exists()]
                parser.error("Incomplete fresh run: missing " + ", ".join(missing)
                             + ". Finish predictions or use --submitted.")
            files = fresh if present else [SUBMITTED / f"{model}.json" for model in MODELS]
            output_dir = args.output_dir or (FRESH if present else Path("runs/recheck"))
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            parser.error("Missing prediction files: " + ", ".join(missing))
        evaluate(files, output_dir, args.bootstrap_repeats, args.seed)


if __name__ == "__main__":
    main()
