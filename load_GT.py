#!/usr/bin/env python3
"""
Convert all MATLAB .mat files (legacy v5 or v7.3 HDF5) under a fixed input directory into grayscale PNGs.
Handles complex-valued data by taking magnitude, collapses any leading dimensions via mean, and preserves directory structure.
"""
from pathlib import Path
import numpy as np
import scipy.io
import h5py
from PIL import Image
import sys

def load_mat(path: Path) -> np.ndarray:
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

def to_real_array(data: np.ndarray) -> np.ndarray:
    # Handle MATLAB complex structured dtype
    if data.dtype.names:
        data = data['real'] + 1j * data['imag']
    if np.iscomplexobj(data):
        data = np.abs(data)
    return data


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255
    return arr.astype(np.uint8)

def collapse_to_2d(data: np.ndarray) -> np.ndarray:
    data = np.squeeze(data)
    data = to_real_array(data)
    if data.ndim == 2:
        return data
    # collapse all dims except the last two via mean
    axes = tuple(range(data.ndim - 2))
    return data.mean(axis=axes)


def convert_and_save(mat_path: Path, output_path: Path) -> None:
    data = load_mat(mat_path)
    arr2d = collapse_to_2d(data)
    if arr2d.ndim != 2:
        raise ValueError(f"Unexpected shape after collapse: {arr2d.shape}")
    img = Image.fromarray(normalize_to_uint8(arr2d))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main():
    input_dir = Path("data/CMR2024_After/groundtruth_mat/MultiCoil/Mapping/ValidationSet/FullSample")
    output_dir =  Path("code/PromptMR-plus/output_chushu/GT_maps")


    print(f"Starting: {Path(__file__).name}")
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    mat_files = list(input_dir.rglob("*.mat"))
    print(f"Found {len(mat_files)} .mat files")

    for mat_file in sorted(mat_files):
        rel = mat_file.relative_to(input_dir)
        png_file = output_dir / rel.with_suffix('.png')
        try:
            convert_and_save(mat_file, png_file)
            print(f"✅ {mat_file} -> {png_file}")
        except Exception as e:
            print(f"✘ Failed {mat_file}: {e}")

if __name__ == "__main__":
    main()
