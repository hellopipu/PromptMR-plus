#!/usr/bin/env python3
"""
Convert all MATLAB .mat files (both v7.3/HDF5 and legacy formats) under a hard‑coded input directory into grayscale PNG images, preserving directory structure.

For multi-dimensional arrays (>2D), collapses all leading dimensions via mean to produce a single 2D slice.
"""
from pathlib import Path
import numpy as np
import scipy.io
import h5py
from PIL import Image


def load_mat(path: Path) -> np.ndarray:
    """Load a .mat file (legacy v5 or HDF5-based v7.3)."""
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"No valid variables in {path}")
            return f[keys[0]][()]
    try:
        mat = scipy.io.loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No valid variables in {path}")
        return mat[keys[0]]
    except Exception as e:
        raise ValueError(f"Cannot load MAT file: {path} ({e})")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255
    return arr.astype(np.uint8)


def collapse_to_2d(data: np.ndarray) -> np.ndarray:
    """Collapse any leading dims by mean, returning a 2D array."""
    data = np.squeeze(data)
    if data.ndim == 2:
        return data
    # collapse all dims except last two
    axes = tuple(range(data.ndim - 2))
    return data.mean(axis=axes)


def convert_and_save(mat_path: Path, output_path: Path) -> None:
    data = load_mat(mat_path)
    arr2d = collapse_to_2d(data)
    if arr2d.ndim != 2:
        raise ValueError(f"Unexpected array shape after collapse: {arr2d.shape}")
    img = Image.fromarray(normalize_to_uint8(arr2d))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main():
    #input_dir = Path("output_chushu/CMR2024/reconstructions/Mapping/ValidationSet/UnderSample_Task1")
    input_dir = Path("data/CMR2024_After/groundtruth_mat/MultiCoil/Mapping/ValidationSet/FullSample")
    output_dir = Path("code/PromptMR-plus/output_chushu/GT_maps")
    for mat_file in sorted(input_dir.rglob("*.mat")):
        rel = mat_file.relative_to(input_dir)
        png_file = output_dir / rel.with_suffix('.png')
        try:
            convert_and_save(mat_file, png_file)
            print(f"✅ {mat_file} -> {png_file}")
        except Exception as e:
            print(f"✘ Failed {mat_file}: {e}")

if __name__ == "__main__":
    main()
