"""Package paper files into arXiv-ready .tar.gz and Overleaf-ready .zip bundles."""

import os
import tarfile
import zipfile
from pathlib import Path

def package():
    repo_root = Path(__file__).resolve().parent.parent
    paper_dir = repo_root / "paper"
    
    files_to_bundle = [
        "main.tex",
        "references.bib",
    ]
    figures = list((paper_dir / "figures").glob("*.png"))
    
    # 1. Overleaf ZIP bundle
    zip_path = paper_dir / "overleaf_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_bundle:
            src = paper_dir / f
            if src.exists():
                zf.write(src, arcname=f)
        for fig in figures:
            zf.write(fig, arcname=f"figures/{fig.name}")
    print(f"Created Overleaf bundle: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    
    # 2. arXiv tar.gz bundle
    tar_path = paper_dir / "arxiv_bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for f in files_to_bundle:
            src = paper_dir / f
            if src.exists():
                tf.add(src, arcname=f)
        for fig in figures:
            tf.add(fig, arcname=f"figures/{fig.name}")
    print(f"Created arXiv bundle: {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    package()
