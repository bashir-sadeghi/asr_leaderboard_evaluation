# Speech recognition on AMI meeting recordings

I compared Whisper Turbo, Whisper large-v3, and FUTO on the same 500 AMI-Cleaned clips. [Read the report](report.html) for methods, findings, and limitations.

## Reproduce the full experiment from audio

From the project folder, with [dependencies installed](#setup), run the commands below (in Colab, add `%%bash` as the first line of the cell).

```bash
python main.py predict --model turbo
python main.py predict --model large-v3
python main.py predict --model futo
python main.py evaluate
```

Predictions and analyses go to `runs/fresh/`. Open `runs/fresh/evaluation.html` for this run. Defaults are 500 clips, seed 17, and batch size 4. The first prediction downloads about 1.85 GB of dataset files plus model weights. To resume, rerun the same command. Use `python main.py predict --help` for device, batch size, and sample size options.

## Check the saved scores

From the project folder, with [dependencies installed](#setup), run:

```bash
python main.py evaluate --submitted
```

This re-scores the included transcripts under nine normalization rules and generates plots and analyses in `runs/recheck/`. Open `runs/recheck/evaluation.html`. It may download the MiniLM text encoder. For the same clips, it reuses the included audio measurements and recorded listening judgments.

As a quick check, the leaderboard rules give word error rates of **9.87% for FUTO, 13.67% for Whisper Turbo, and 14.12% for Whisper large-v3**.

## Setup

Use Python 3.12 or 3.13 and run commands from the project folder. Install matching PyTorch and TorchAudio for your hardware, then install `requirements.txt`. The full workflow was verified on a Colab T4 with CUDA, Python 3.13, PyTorch/TorchAudio 2.8.0+cu126, and Transformers 4.57.3. That Colab runtime needed its incompatible preinstalled TorchCodec removed and a restart after changing PyTorch.

For a local CPU environment:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install -r requirements.txt
```

On Linux, add `--index-url https://download.pytorch.org/whl/cpu` to the first command for CPU wheels. The earlier Mac FUTO CPU run used Torch 2.3.0.
