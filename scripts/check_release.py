"""Fail the image build if any packaged model or data artifact is missing/changed."""
import hashlib
import json
from pathlib import Path


def check(root=None):
    root = root or Path(__file__).resolve().parents[1]
    assets = root / "deploy/assets"
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (assets / name).resolve()
        if not path.is_relative_to(assets.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release asset failed verification: {name}")
    print(f"Verified {len(manifest['files'])} release assets.")


if __name__ == "__main__":
    check()
