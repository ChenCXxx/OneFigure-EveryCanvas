# Flowchart Benchmark Usage Guide

This document covers environment setup, input format, how to run the
benchmark, output format, and how to select which benchmarks to run.

## 1. Environment Setup

Python 3.12 with `uv` is recommended.

### Windows

```powershell
cd C:\Users\User\Documents\lab\rebuttal\benchmark

uv venv --python 3.12
uv sync --extra space

Copy-Item .env.example .env
```

### Linux

```bash
cd /path/to/rebuttal/benchmark

uv venv --python 3.12
uv sync --extra space

cp .env.example .env
```

In `.env`, configure which model provider to use, for example Gemini:

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-pro
GEMINI_API_KEY=your-api-key
```

Do not commit `.env` or your API key.

## 2. Input File Structure

Input data goes under `benchmark/inputs/`, with one folder per case:

```text
benchmark/
└─ inputs/
   ├─ CVPR2025-00006-1_1_1/
   │  ├─ reference.jpeg
   │  ├─ af.png
   │  ├─ gpt.png
   │  ├─ nano-banana.png
   │  ├─ ours.png
   │  └─ paperbanana.png
   └─ CVPR2025-00006-1_2_3/
      ├─ reference.jpeg
      ├─ af.png
      ├─ gpt.png
      ├─ nanobanana.png
      ├─ ours.png
      └─ paperbanana.png
```

Rules:

- Folder names follow the format `<case_id>_<width>_<height>`.
- `reference.png`, `reference.jpg`, `reference.jpeg`, `reference.webp`, or
  `reference.bmp` is the reference image.
- Any other `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` image is treated as a
  candidate.
- A candidate's filename becomes its method name, e.g. `ours.png` becomes
  `ours`.

## 3. Running the Benchmark

Run from the `benchmark/` directory:

```powershell
uv run python pipeline.py `
    --input-dir .\inputs `
    --output-dir .\outputs `
    --name run1
```

Linux:

```bash
uv run python pipeline.py \
    --input-dir ./inputs \
    --output-dir ./outputs \
    --name run1
```

If `--benchmarks` is not specified, the following run by default:

```text
style → space → relationship → hallucination
```

## 4. Selecting Which Benchmarks to Run

Use `--benchmarks` to select one or more benchmarks.

Run only Style:

```powershell
uv run python pipeline.py --benchmarks style
```

Run Style and Space:

```powershell
uv run python pipeline.py --benchmarks style space
```

Run Relationship and Hallucination:

```powershell
uv run python pipeline.py --benchmarks relationship hallucination
```

Run all four:

```powershell
uv run python pipeline.py `
    --benchmarks style space relationship hallucination
```

Available benchmark names:

```text
style
space
relationship
hallucination
```

## 5. Reusing Space Metrics

First run:

```powershell
uv run python pipeline.py `
    --name run1
```

A later run can reuse Space metrics from `run1` that are still unchanged:

```powershell
uv run python pipeline.py `
    --name run2 `
    --reuse-metrics `
    --previous-root .\outputs\run1
```

Metrics are only reused when the case, method, reference image, and
candidate image are all identical. If an image has changed, its metrics are
recomputed.

## 6. Output Structure

```text
benchmark/
└─ outputs/
   └─ run1/
      ├─ run.json
      └─ <case_key>/
         ├─ style/
         │  └─ result.json
         ├─ space/
         │  ├─ result.json
         │  ├─ metrics/
         │  │  └─ <method>.json
         │  └─ artifacts/
         │     └─ <method>_empty_regions.png
         ├─ relationship/
         │  └─ result.json
         └─ hallucination/
            └─ result.json
```

`run.json` is a summary of the entire run, containing:

- run id
- benchmarks executed
- provider and model
- run status
- results for each case

`space/metrics/` stores Space's image analysis results, and
`space/artifacts/` stores visualizations of blank regions.

While running, the terminal shows progress for the current case, benchmark,
and method.
