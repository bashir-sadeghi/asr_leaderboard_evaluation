# Speech recognition on AMI meeting recordings

I compared Whisper Turbo, Whisper large-v3, and FUTO on the same 500 AMI-Cleaned clips.

- **Main report:** [Three-page PDF](report.pdf) · [HTML](report.html).
- **Full report (optional):** [PDF](report_full.pdf) · [Interactive HTML](report_full.html).

## Reproduce the full experiment from audio

From the project folder, with [dependencies installed](#setup), run the commands below (in Colab, add `%%bash` as the first line of the cell).

```bash
python main.py predict --model turbo
python main.py predict --model large-v3
python main.py predict --model futo
python main.py evaluate
```

The commands create `runs/fresh/` automatically and save predictions and analyses there. Open `runs/fresh/evaluation.html` for this run. Defaults are 500 clips, seed 17, and batch size 4. Downloads are about 1.85 GB for the dataset, plus model weights; cached files are reused. To resume, rerun the same command with the same settings. Use `python main.py predict --help` for device, batch size, and sample size options.

## Check the saved scores

From the project folder, with [dependencies installed](#setup), run:

```bash
python main.py evaluate --submitted
```

This re-scores the included transcripts under nine normalization rules and creates `runs/recheck/` automatically for the plots and analyses. Open `runs/recheck/evaluation.html`. It may download the MiniLM text encoder. For the same clips, it reuses the included audio measurements and recorded listening judgments.

As a quick check, the leaderboard rules give word error rates of **9.87% for FUTO, 13.67% for Whisper Turbo, and 14.12% for Whisper large-v3**.

## Setup

Use Python 3.12 or 3.13 and run commands from the project folder. Choose one setup below. The full workflow was verified on a Colab T4 with CUDA, Python 3.13, PyTorch/TorchAudio 2.8.0+cu126, and Transformers 4.57.3. That Colab runtime needed its incompatible preinstalled TorchCodec removed and a restart after changing PyTorch.

### macOS on Apple Silicon (CPU)

Install everything with one command:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 -r requirements.txt
```

### Linux (CPU)

Install the CPU builds from PyTorch's package index, then the remaining dependencies from PyPI:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### Linux (CUDA)

Requires an NVIDIA GPU and a compatible NVIDIA driver. Confirm that this command lists your GPU before installing the Python packages:

```bash
nvidia-smi
```

Install the CUDA 12.6 builds and the remaining dependencies:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

To require GPU execution:

```bash
python main.py predict --model turbo --device cuda
```

PyTorch supplies the CUDA runtime libraries; a separate CUDA Toolkit installation is unnecessary for this workflow. [PyTorch explanation](https://discuss.pytorch.org/t/about-cuda-cudnn/182963/2)

Without an explicit device, prediction uses CUDA when available, otherwise CPU. MPS is not supported. Package versions follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v280). The earlier Mac FUTO CPU run used Torch 2.3.0; the 2.8.0 Mac CPU setup has not been run end to end.
