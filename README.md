# One Figure, Every Canvas

## Overview


## Download and Installation

```bash
git clone <repository-url>
cd OneFigure-EveryCanvas

uv venv --python 3.12
uv sync --all-packages --all-extras
```

Create the local environment files and add the required API keys:

```bash
cp pipeline/.env.example pipeline/.env
cp benchmark/.env.example benchmark/.env
```

Do not commit `.env` files.

## Repository Structure

```text
.
├── pipeline/
│   ├── inputs/
│   ├── outputs/
│   ├── pipeline.py
│   ├── prompts/
│   └── tools/
├── benchmark/
│   ├── inputs/
│   ├── outputs/
│   ├── benchmarks/
│   └── prompts/
├── pyproject.toml
└── uv.lock
```

## Pipeline

The pipeline runs the parse, style, and layout stages to generate a flowchart.

### Input and Output

The input directory must contain `figure.png`.

```text
pipeline/
├── inputs/
│   └── <case_name>/
│       └── figure.png
└── outputs/
    └── <run_name>/
        └── <image_name>/
```

### Run

```bash
cd pipeline

uv run python pipeline.py \
  --input_dir inputs/<case_name> \
  --output_dir outputs/ \
  --image_name <image_name> \
  --aspect_ratio 16:9
```

`--output_dir` specifies the output root directory, while the required
`--image_name` specifies its subdirectory name. The output path is therefore
`outputs/<image_name>/`.

`--aspect_ratio` is also optional and accepts formats such as `16:9`, `4:3`,
or `1:1`. If it is omitted, the layout preserves the original canvas size and
aspect ratio obtained from the input image.

### SAM3 Setup

SAM3 is required when bounding boxes are not provided. The pipeline expects the
official SAM3 repository at `pipeline/sam3/`:

```bash
cd pipeline
git clone https://github.com/facebookresearch/sam3.git
uv pip install -e ./sam3
cd ..
```

Request access to the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3)
checkpoint, then authenticate with Hugging Face:

```bash
hf auth login
```

The checkpoint is downloaded automatically when SAM3 first loads the model.
See the [official SAM3 repository](https://github.com/facebookresearch/sam3)
for the complete setup and license information.

## Benchmark

The benchmark evaluates reference images and candidate flowchart images.

### Input and Output

Each case is stored in its own directory. The reference image filename must
start with `reference`; other image files are treated as candidates.

```text
benchmark/
├── inputs/
│   └── <case_name>/
│       ├── reference.png
│       ├── ours.png
│       └── other_method.png
└── outputs/
    └── <run_name>/
```

### Run

```bash
cd benchmark

uv run python pipeline.py \
  --input-dir ./inputs \
  --output-dir ./outputs \
  --name <run_name>
```

If `--benchmarks` is omitted, the default is to run all four benchmarks:
`style`, `space`, `relationship`, and `hallucination`.

To run only the Space benchmark:

```bash
uv run python pipeline.py \
  --input-dir ./inputs \
  --output-dir ./outputs \
  --name <run_name> \
  --benchmarks space
```

### Optional Arguments

- `--input-dir`: input root directory; defaults to `./inputs`.
- `--output-dir`: output root directory; defaults to `./outputs`.
- `--name`: run directory name; defaults to the current timestamp.
- `--benchmarks`: one or more of `style`, `space`, `relationship`, and
  `hallucination`. If omitted, all four are run.
