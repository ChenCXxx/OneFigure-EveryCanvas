# One Figure, Every Canvas
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

## 📌 Description


## 📂 Repository Structure

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

## 🚀 Getting Started

### Installation

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

### Pipeline

The pipeline runs the parse, style, and layout stages to generate a flowchart.

<details>
<summary>draw.io CLI Setup</summary>

The pipeline uses the draw.io desktop CLI to render intermediate and final XML
files as images.

#### Windows

Download and install the [draw.io Windows installer](https://github.com/jgraph/drawio-desktop/releases/download/v31.7.0/draw.io-31.7.0-windows-installer.exe).
If draw.io is installed in the default location, add the following to
`pipeline/.env`:

```env
DRAWIO_BIN=C:/Program Files/draw.io/draw.io.exe
```

#### Linux

For an x86_64 Linux machine, download the AppImage:

```bash
cd /tmp
wget https://github.com/jgraph/drawio-desktop/releases/download/v31.7.0/drawio-x86_64-31.7.0.AppImage
chmod +x drawio-x86_64-31.7.0.AppImage
mkdir -p ~/.local/bin
cp drawio-x86_64-31.7.0.AppImage ~/.local/bin/drawio
export PATH="$HOME/.local/bin:$PATH"
```

If `~/.local/bin` is not already on `PATH`, set the executable explicitly in
`pipeline/.env`:

```env
DRAWIO_BIN=/home/<username>/.local/bin/drawio
```

On a Linux desktop with a graphical session, run the pipeline normally. On a
headless Linux server, install `xvfb` and run the pipeline through
`xvfb-run`:

```bash
sudo apt install -y xvfb

xvfb-run -a \
  --server-args="-screen 0 1280x1024x24" \
  uv run python pipeline.py \
  --input_dir inputs/<case_name> \
  --output_dir outputs/ \
  --image_name <image_name> \
  --aspect_ratio 16:9
```

</details>

<details>
<summary>SAM3 Setup</summary>

SAM3 is required when bounding boxes are not provided. The pipeline expects the
official SAM3 repository at `pipeline/sam3/`:

```bash
cd pipeline
git clone https://github.com/facebookresearch/sam3.git
uv pip install einops ninja pycocotools
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

</details>

#### Input and Output

The input directory must contain `figure.png`.

```text
pipeline/
├── inputs/
│   └── <case_name>/
│       └── figure.png
└── outputs/
    └── <image_name>/
        ├── parse/
        │   ├── iter_1/
        │   │   ├── parse.xml
        │   │   ├── parse.jpg
        │   │   └── critic.json
        │   ├── bbox/
        │   │   └── bbox.json
        │   └── final.xml
        ├── style/
        │   ├── iter_1/
        │   │   ├── style.xml
        │   │   ├── style.png
        │   │   └── critic.json
        │   └── final.xml
        ├── layout/
        │   ├── iter_1/
        │   │   ├── layout.xml
        │   │   ├── layout.png
        │   │   └── critic.json
        │   └── final.xml
        ├── image/
        │   ├── crop_image/
        │   │   └── <cell_id>.png
        │   ├── final.xml
        │   └── final.png
        ├── run.json
        └── timeline.json
```

The `iter_*` directories contain intermediate results for each iteration.
The main stage outputs are `parse/final.xml`, `style/final.xml`, and
`layout/final.xml`. The final self-contained diagram is
`image/final.png`, with its corresponding XML at `image/final.xml`.

- `parse/`: parses the input image into diagram XML and generates bounding-box
  data when needed.
- `style/`: applies visual styling to the parsed diagram.
- `layout/`: arranges the diagram on the target canvas.
- `image/`: crops and embeds image elements, then renders the final PNG.
- `run.json` and `timeline.json`: store run configuration, output paths, and
  stage timing information.

#### Command

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

### Benchmark

The benchmark evaluates reference images and candidate flowchart images.

#### Input and Output

Each case is stored in its own directory. The reference image filename must
start with `reference`; other image files are treated as candidates.

```text
benchmark/
├── inputs/
│   └── <case_id>/
│       ├── reference.png
│       ├── ours.png
│       └── other_method.png
└── outputs/
    └── <run_name>/
        ├── run.json
        └── <case_id>/
            ├── style/
            │   └── result.json
            ├── space/
            │   ├── result.json
            │   ├── metrics/
            │   │   └── <method>.json
            │   └── artifacts/
            │       └── <method>_empty_regions.png
            ├── relationship/
            │   └── result.json
            └── hallucination/
                └── result.json
```

- `run.json`: summary of the complete benchmark run.
- `<case_id>/`: results for one input case.
- `result.json`: benchmark result and status for the corresponding task.
- `space/metrics/`: deterministic Space metrics for each candidate method.
- `space/artifacts/`: visual diagnostic files generated by the Space benchmark.

#### Command

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

#### Optional Arguments

- `--input-dir`: input root directory; defaults to `./inputs`.
- `--output-dir`: output root directory; defaults to `./outputs`.
- `--name`: run directory name; defaults to the current timestamp.
- `--benchmarks`: one or more of `style`, `space`, `relationship`, and
  `hallucination`. If omitted, all four are run.

## 📊 Results


## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{onefigureeverycanvas2026,
  title={One Figure, Every Canvas: Editable Flowchart Relayout via Agentic Pipeline},
  author={Shih-Chen Tseng and Chih-Hsuan Chen and Ryan Yang and Hsi-An Chen and Chun-Wei Tuan Mu and Yu-Lun Liu},
  year={2026}
}
```

## 📄 License
This project is licensed under the [MIT License](LICENSE).
