import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _describe(value, indent="    "):
    if isinstance(value, dict):
        print(f"{indent}(dict, {len(value)} key(s)): {list(value.keys())[:10]}")
    elif hasattr(value, "__len__") and not isinstance(value, str):
        arr = np.asarray(value)
        print(f"{indent}shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim == 1 and arr.size:
            sample = arr[: min(10, arr.size)]
            print(f"{indent}first values: {sample.tolist()}")
            if np.issubdtype(arr.dtype, np.integer):
                vals, counts = np.unique(arr, return_counts=True)
                print(f"{indent}unique values + counts: {dict(zip(vals.tolist(), counts.tolist()))}")
    else:
        print(f"{indent}{value!r}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    print(f"loading {path} (large split files can take a minute and real memory)...")
    with open(path) as f:
        data = json.load(f)

    print(f"\ntop-level keys ({len(data)}): {list(data.keys())}")
    for top_key, sequences in data.items():
        if not isinstance(sequences, dict):
            print(
                f"\n{top_key!r}: not a dict ({type(sequences).__name__}) - calms21_import.py's iter_sequences skips this"
            )
            continue
        seq_names = list(sequences.keys())
        print(f"\n{top_key!r}: {len(seq_names)} sequence(s)")
        print(f"  first few sequence names: {seq_names[:5]}")
        name, seq = next(iter(sequences.items()))
        print(f"\n  inspecting one sequence ({name!r}):")
        for k, v in seq.items():
            print(f"  {k}:")
            _describe(v)
        break

    print(
        "\nCompare the shapes/keys above to pipeline/calms21_import.py's module "
        "docstring (keypoints should be (n_frames, 2, 2, 7), scores (n_frames, 2, 7), "
        "annotations (n_frames,) with values 0-3). If anything differs, say so before "
        "running a real conversion - the module needs a matching fix first."
    )


if __name__ == "__main__":
    main()
