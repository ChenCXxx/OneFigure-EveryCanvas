<h1 align="center">One Figure, Every Canvas: Editable Flowchart Relayout via Agentic Pipeline</h1>

<p align="center">
  <a href="https://onefigureeverycanvas.vercel.app/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2610.06852"><img src="https://img.shields.io/badge/arXiv-2610.06852-b31b1b" alt="arXiv"></a>
  <a href="https://drive.google.com/file/d/1WH9etIssytkrmkKSI1_mhNoTy9FEkRjk/view"><img src="https://img.shields.io/badge/Dataset-Google%20Drive-blue" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/%E5%A3%AB%E7%8F%8D-%E6%9B%BE-125166331/"><strong>Shih-Chen Tseng</strong></a><sup>1,*</sup>,
  <a href="https://www.linkedin.com/in/chchen825/"><strong>Chih-Hsuan Chen</strong></a><sup>1,*</sup>,
  <a href="https://www.linkedin.com/in/rhy01/"><strong>Ryan Yang</strong></a><sup>2,*</sup>,
  <a href="https://www.linkedin.com/in/annchen1234/"><strong>Hsi-An Chen</strong></a><sup>1</sup>,
  <a href="https://www.linkedin.com/in/ray-tuan-mu-a46257246/"><strong>Chun-Wei Tuan Mu</strong></a><sup>1</sup>,
  <a href="https://yulunalexliu.github.io/"><strong>Yu-Lun Liu</strong></a><sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup> National Yang Ming Chiao Tung University &nbsp; <sup>2</sup> University of Illinois at Urbana-Champaign<br>
  <sub>* Equal contribution &nbsp; † Corresponding author</sub>
</p>

## Key Idea

Given a single raster flowchart and a target aspect ratio, we adapt the figure to a new canvas while preserving its content and connections. Our **Parse, Style, and Layout agents** each work with a critic to check structure, appearance, and layout. The output is **fully editable in draw.io**, making the same figure reusable across papers, slides, posters, and phone previews.

## Outline

- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Pipeline](#pipeline)
  - [Benchmark](#benchmark)
- [Dataset](#dataset)
- [Citation](#citation)
- [License](#license)

<a id="repository-structure"></a>

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
├── requirements.txt
├── pyproject.toml
└── uv.lock
```

<a id="quick-start"></a>

## Quick Start

### Installation

```bash
git clone <repository-url>
cd OneFigure-EveryCanvas

uv venv --python 3.12
uv sync --all-packages --all-extras
```

If `uv` is not available, use `venv` and `pip` instead:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
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
python -m pip install einops ninja pycocotools
python -m pip install -e ./sam3
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

#### Iterations

`--parse_iterations`, `--style_iterations`, and `--layout_iterations` set
the **maximum iterations** per stage (default: **1** each). Stages may stop
early when the critic accepts the result.

Our paper uses **Parse: 3, Style: 2, Layout: 5**. From `pipeline/`:

```bash
uv run python pipeline.py \
  --input_dir inputs/<case_name> \
  --output_dir outputs/ \
  --image_name <image_name> \
  --aspect_ratio 16:9 \
  --parse_iterations 3 \
  --style_iterations 2 \
  --layout_iterations 5
```

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

## Dataset

We release **FlowchartRelayoutBench**, containing 100 flowcharts collected from oral papers and relayout results from **our method and all evaluated baselines** at five target aspect ratios: **9:16, 2:3, 1:1, 3:2, and 16:9**. The dataset includes the original flowcharts and generated images, allowing researchers to run our benchmark and compare their own methods against the provided results.

**[Download the dataset (Google Drive)](https://drive.google.com/file/d/1WH9etIssytkrmkKSI1_mhNoTy9FEkRjk/view)**

### Dataset Structure

```text
flowchart-relayout-bench-dataset/
└── cases/
    ├── case_0001/
    │   ├── source.png
    │   └── relayouts/
    │       ├── 1_1/
    │       │   ├── autofigure_edit.png
    │       │   ├── gpt_image_2.png
    │       │   ├── nano_banana_pro.png
    │       │   ├── ours.png
    │       │   └── paperbanana.png
    │       ├── 2_3/
    │       ├── 3_2/
    │       ├── 9_16/
    │       └── 16_9/
    ├── case_0002/
    ├── ...
    └── case_0100/
```

`source.png` is the original flowchart. Each folder under `relayouts/` stores the output images for one target aspect ratio; for example, `9_16` represents 9:16. Method filenames identify our result (`ours.png`) and each baseline.

### Benchmark the Dataset

To use the [benchmark pipeline](#benchmark), prepare a separate input folder for each case and target ratio. Copy the original `source.png` into that folder as `reference.png`, along with the method images from the corresponding `relayouts/<ratio>/` folder. Add your own method's image to the same folder to include it in the comparison.

For example, prepare `case_0001` at 1:1 as:

```text
benchmark/inputs/case_0001_1_1/
├── reference.png          # copied from case_0001/source.png
├── autofigure_edit.png
├── gpt_image_2.png
├── nano_banana_pro.png
├── ours.png
└── paperbanana.png
```

Repeat for the cases and ratios you want to evaluate, then follow the [benchmark command](#benchmark). The released dataset structure must be converted to this input format before running the benchmark.

<a id="citation"></a>

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{onefigureeverycanvas2026,
  title={One Figure, Every Canvas: Editable Flowchart Relayout via Agentic Pipeline},
  author={Shih-Chen Tseng and Chih-Hsuan Chen and Ryan Yang and Hsi-An Chen and Chun-Wei Tuan Mu and Yu-Lun Liu},
  year={2026}
}
```

<a id="license"></a>

## License
This project is licensed under the [MIT License](LICENSE).
