#!/usr/bin/env python3
"""
Reads an Excel file (default: pod_list.xlsx, sheet: Sheet1) and prints
"<namespace>/<name>" for every row.

If the required Python packages (pandas, openpyxl) are missing, the
script installs them automatically using pip.
"""
import sys
import subprocess
import importlib
from typing import List


def ensure_packages(packages: List[str]) -> None:
    """Ensure all packages in *packages* are importable; install them if not."""
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"[INFO] Installing missing package: {pkg}", file=sys.stderr)
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def main() -> None:
    required = ["pandas", "openpyxl"]
    ensure_packages(required)

    import pandas as pd  # noqa: E402  # import after ensuring package exists
    import argparse  # noqa: E402

    parser = argparse.ArgumentParser(
        description="Print <namespace>/<name> for each row in an Excel sheet")
    parser.add_argument(
        "--file", "-f", default="pod_list.xlsx", help="Path to the Excel file")
    parser.add_argument(
        "--sheet", "-s", default="Sheet1", help="Sheet name to read")
    args = parser.parse_args()

    try:
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Standardize column names: strip whitespace, convert to uppercase
    df.columns = [str(col).strip().upper() for col in df.columns]

    missing_cols = [col for col in ("NAMESPACE", "NAME") if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required column(s): {', '.join(missing_cols)}", file=sys.stderr)
        sys.exit(1)

    for namespace, name in zip(df["NAMESPACE"], df["NAME"]):
        if pd.isna(namespace) or pd.isna(name):
            # Skip rows with missing values
            continue
        print(f"{namespace}/{name}")


if __name__ == "__main__":
    main()