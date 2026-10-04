from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import convert_yolo_bbox_txt_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import bbox.txt files by case name and generate missing bbox.json files."
        )
    )
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument(
        "--overwrite_json",
        action="store_true",
        help="Regenerate bbox.json even when it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root not found: {source_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root not found: {input_root}")

    source_by_case: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for bbox_txt in source_root.rglob("bbox.txt"):
        case_name = bbox_txt.parent.name
        previous = source_by_case.get(case_name)
        if previous is None:
            source_by_case[case_name] = bbox_txt
        else:
            duplicates.setdefault(case_name, [previous]).append(bbox_txt)
    if duplicates:
        details = "\n".join(
            f"{case}: {paths}" for case, paths in sorted(duplicates.items())
        )
        raise RuntimeError(f"Duplicate source case names:\n{details}")

    imported = 0
    unchanged = 0
    converted = 0
    preserved_json = 0
    missing_source: list[str] = []

    for case_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        case_name = case_dir.name
        source_txt = source_by_case.get(case_name)
        if source_txt is None:
            missing_source.append(case_name)
            print(f"[bbox-import][{case_name}] no source; unchanged", flush=True)
            continue

        destination_txt = case_dir / "bbox.txt"
        destination_json = case_dir / "bbox.json"
        image_path = case_dir / "figure.png"
        txt_was_same = (
            destination_txt.is_file()
            and destination_txt.read_bytes() == source_txt.read_bytes()
        )

        if txt_was_same:
            unchanged += 1
            print(f"[bbox-import][{case_name}] bbox.txt already matches", flush=True)
        else:
            shutil.copy2(source_txt, destination_txt)
            imported += 1
            print(
                f"[bbox-import][{case_name}] copied {source_txt} -> "
                f"{destination_txt}",
                flush=True,
            )

        if destination_json.is_file() and not args.overwrite_json:
            preserved_json += 1
            print(
                f"[bbox-import][{case_name}] preserving existing bbox.json",
                flush=True,
            )
            continue

        convert_yolo_bbox_txt_to_json(
            bbox_txt_path=destination_txt,
            image_path=image_path,
            output_json_path=destination_json,
        )
        converted += 1

    print(
        "[bbox-import] completed "
        f"copied={imported} unchanged_txt={unchanged} "
        f"converted_json={converted} preserved_json={preserved_json} "
        f"missing_source={len(missing_source)}",
        flush=True,
    )
    if missing_source:
        print(
            "[bbox-import] cases without source: " + ", ".join(missing_source),
            flush=True,
        )


if __name__ == "__main__":
    main()
