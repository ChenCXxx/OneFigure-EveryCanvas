# Flowchart Benchmark 使用說明

這份文件提供 benchmark 的環境設定、輸入格式、執行方式、輸出格式與
benchmark 選擇方式。

## 1. 環境設定

建議使用 Python 3.12 與 `uv`。

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

在 `.env` 中設定要使用的模型服務，例如 Gemini：

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-pro
GEMINI_API_KEY=your-api-key
```

請勿提交 `.env` 或 API key。

## 2. 輸入檔案架構

輸入資料放在 `benchmark/inputs/` 底下，每個 case 使用一個資料夾：

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

規則：

- 資料夾名稱格式為 `<case_id>_<width>_<height>`。
- `reference.png`、`reference.jpg`、`reference.jpeg`、`reference.webp` 或
  `reference.bmp` 是參考圖片。
- 其他 `.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp` 圖片會被視為 candidate。
- candidate 的檔名會成為 method 名稱，例如 `ours.png` 會得到 `ours`。

## 3. 執行 benchmark

在 `benchmark/` 目錄執行：

```powershell
uv run python pipeline.py `
    --input-dir .\inputs `
    --output-dir .\outputs `
    --name run1
```

Linux：

```bash
uv run python pipeline.py \
    --input-dir ./inputs \
    --output-dir ./outputs \
    --name run1
```

如果沒有指定 `--benchmarks`，預設執行：

```text
style → space → relationship → hallucination
```

## 4. 選擇要執行的 benchmark

可使用 `--benchmarks` 選擇一個或多個 benchmark。

只執行 Style：

```powershell
uv run python pipeline.py --benchmarks style
```

執行 Style 與 Space：

```powershell
uv run python pipeline.py --benchmarks style space
```

執行 Relationship 與 Hallucination：

```powershell
uv run python pipeline.py --benchmarks relationship hallucination
```

執行全部四個：

```powershell
uv run python pipeline.py `
    --benchmarks style space relationship hallucination
```

可使用的 benchmark 名稱：

```text
style
space
relationship
hallucination
```

## 5. 重用 Space metrics

第一次執行：

```powershell
uv run python pipeline.py `
    --name run1
```

之後的執行可以重用 `run1` 中仍然相同的 Space metrics：

```powershell
uv run python pipeline.py `
    --name run2 `
    --reuse-metrics `
    --previous-root .\outputs\run1
```

只有在 case、method、reference 圖片與 candidate 圖片都相同時才會重用。
如果圖片有變更，對應的 metrics 會重新計算。

## 6. Output 結構

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

`run.json` 是整次執行的摘要，包含：

- run id
- 執行的 benchmark
- provider 與 model
- 執行狀態
- 每個 case 的結果

`space/metrics/` 存放 Space 的圖片分析結果，
`space/artifacts/` 存放空白區域的視覺化圖片。

執行完成後，terminal 會顯示目前的 case、benchmark 與 method 進度。
