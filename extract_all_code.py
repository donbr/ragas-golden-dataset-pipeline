#!/usr/bin/env python3
"""
Recursively extract and combine all Python code from a directory into a single file.

Usage:
    extract_all_code.py [-n] output_file [input_directory]

Options:
    -n, --dry-run    List files to be combined without writing output
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set

# Directories to exclude from traversal
EXCLUDED_DIRS: Set[str] = {
    '.git', '.venv', 'venv', '__pycache__',
    'node_modules', '.pytest_cache', '.tox',
    '.mypy_cache', '.ruff_cache', '.egg-info',
    'dist', 'build', '.ipynb_checkpoints',
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine all .py files from a directory into one file."
    )
    parser.add_argument(
        'output',
        type=Path,
        help='Path to the combined output .py file'
    )
    parser.add_argument(
        'source',
        type=Path,
        nargs='?',
        default=Path.cwd(),
        help='Directory to search for .py files (default: current directory)'
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='List files without creating the output file'
    )
    return parser.parse_args()


def find_python_files(base: Path, excluded: Set[str], script: Path) -> List[Path]:
    files: List[Path] = []
    for path in base.rglob('*.py'):
        # skip this script
        if path.resolve() == script.resolve():
            continue
        # skip excluded directories
        if any(part in excluded for part in path.parts):
            continue
        files.append(path)
    # sort by directory depth then name
    return sorted(files, key=lambda p: (len(p.parts), p.as_posix()))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log = logging.getLogger()

    source_dir = args.source.resolve()
    output_file = args.output
    script_path = Path(__file__).resolve()

    if not source_dir.is_dir():
        log.error("Source directory '%s' does not exist or is not a directory.", source_dir)
        sys.exit(1)

    python_files = find_python_files(source_dir, EXCLUDED_DIRS, script_path)

    if not python_files:
        log.info("No Python files found in %s", source_dir)
        sys.exit(0)

    log.info("Found %d Python files in %s", len(python_files), source_dir)

    if args.dry_run:
        for f in python_files:
            log.info(f.as_posix())
        sys.exit(0)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"#!/usr/bin/env python3\n"
        f"# Combined on {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"# Source directory: {source_dir}\n"
        f"# Excluded directories: {', '.join(sorted(EXCLUDED_DIRS))}\n\n"
    )

    try:
        with output_file.open('w', encoding='utf-8') as out:
            out.write(header)
            last_dir = None
            for path in python_files:
                rel = path.relative_to(source_dir)
                parent = rel.parent.as_posix() or '.'
                if parent != last_dir:
                    out.write(f"\n# {'='*10} Directory: {parent} {'='*10}\n")
                    last_dir = parent
                out.write(f"\n# {'-'*40}\n# File: {rel}\n# {'-'*40}\n")
                for line in path.open('r', encoding='utf-8'):
                    out.write(line)
        log.info("Combined code written to %s", output_file)
    except Exception as e:
        log.error("Failed to write output: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
