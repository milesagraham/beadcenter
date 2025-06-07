import os
import glob
import mrcfile
import pandas as pd

import numpy as np
from imodmodel.models import ContourType
from scipy.ndimage import label, center_of_mass, gaussian_filter
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu

from typing import Tuple
import imodmodel
from pydantic import BaseModel

import matplotlib

matplotlib.use('TkAgg')  # Or 'QtAgg' if you have Qt installed
import matplotlib.pyplot as plt

# ===CONFIGURATION ===

BLUR_SIGMA = 5
PATCH_SIZE = 80
BACKUP_SUFFIX = "_backup"

class Coord(BaseModel):
    center: Tuple[float, float, float]


def extract_fiducial_points(fid_path):
    """
    Reads the existing mod file and returns a dataframe with points grouped by contour

    Parameters:
        the .fid file from IMOD to be read

    Returns:
        Dataframe with the .fid file information.
    """
    df = imodmodel.read(fid_path)
    grouped = []

    for (object_id, contour_id), group in df.groupby(['object_id', 'contour_id']):
        points = []
        for _, row in group.iterrows():
            points.append(row)
        grouped.append((object_id, contour_id, points))

    return grouped


def extract_subregion(stack_path, coord, size):
    """
    Extracts a patch from the image stack around each coordinate.

    Parameters:
        stack_path : Input image stack.
        coord: the existing coordinate from the mod file
        size: the size of 2D image to extract in which to search for the bead center

    Returns:
        A 2d image patch and the xyz information of the 2d image patch.
    """
    x, y, z = coord
    x, y, z = int(round(x)), int(round(y)), int(round(z))
    half = size // 2

    with mrcfile.open(stack_path, permissive=True) as mrc:
        image = mrc.data[z]
        height, width = image.shape

        x_min = max(0, x - half)
        y_min = max(0, y - half)
        x_max = min(width, x + half)
        y_max = min(height, y + half)
        patch = image[y_min:y_max, x_min:x_max]
        patch = patch.astype(np.float32)

    return patch, x_min, y_min, z


def apply_gaussian_smoothing(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 2D image.

    Parameters:
        image (np.ndarray): Input 2D image.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: Smoothed image.
    """
    assert image.ndim == 2, "Only 2D grayscale images supported"
    filtered = gaussian_filter(image, sigma=sigma)
    inverted = np.max(filtered) - filtered

    return inverted


def mask_beads(patch):
    """
    Creates a binary mask for the gold bead using automatic thresholding

    Parameters:
        image (np.ndarray): Input 2D image.

    Returns:
        np.ndarray: Mask.
    """
    thresh_val = threshold_otsu(patch)
    higher_than_threshold = patch > thresh_val
    lower_than_threshold = patch < thresh_val
    patch[higher_than_threshold] = 1
    patch[lower_than_threshold] = 0
    # Remove small noise regions
    mask = remove_small_objects(patch.astype(bool), min_size=3)
    return mask


def shrink_patch_size(patch):
    """
    If multiple beads are detected in the image, then the patch size is shrunk until only one remains.

    Parameters:
        image (np.ndarray): Input 2D image.

    Returns:
        np.ndarray: cropped patch and it's shape
    """
    size = patch.shape[0]
    crop = int(size * 0.05 / 2)  # 2.5% from each side

    # Ensure we don't crop too much
    if size - 2 * crop < 4:
        raise ValueError("Patch too small to shrink further.")

    cropped = patch[crop:-crop, crop:-crop]
    return cropped, cropped.shape[0]


def detect_blob_center(patch):
    """
    Finds the center of the gold bead from the input mask.

    Parameters:
        image (np.ndarray): Input 2D mask.

    Returns:
        np.ndarray: coordinates of the new center
        updated patch size and patch for diagnostic use.
    """
    patch_size = patch.shape[0]
    mask = mask_beads(patch)
    # Label connected components
    labeled, num_features = label(mask)

    if num_features == 0:
        raise ValueError("No blob found in the image.")

    while num_features > 1:
        patch, patch_size = shrink_patch_size(patch)
        mask = mask_beads(patch)
        labeled, num_features = label(mask)

    # Get the largest region
    largest_region = (labeled == np.argmax(np.bincount(labeled.flat)[1:]) + 1)

    # Compute center of mass of that region
    cy, cx = center_of_mass(largest_region)

    return cx, cy, patch_size, patch

def matplotlib_diagnostics(patch, cx, cy):

    """
    Plotting function for diagnostic purposes
    """
    plt.imshow(patch, cmap='gray')
    plt.scatter(cx, cy, color='red', marker='x')
    plt.title("Detected Bead Center")
    plt.show()


def refine_and_save_fiducials(fid_path, stack_path, backup_suffix, patch_size, blur_sigma):
    """
    Runs the required image processing and centering functions and converts the output back into full image dimension coordinates.
    The new coordinates are then saved into a new .fid file as an open contour.
    """
    contour_groups = extract_fiducial_points(fid_path)
    refined_rows = []

    for object_id, contour_id, rows in contour_groups:
        for row in rows:
            x, y, z = row["x"], row["y"], row["z"]
            try:
                patch, x_min, y_min, _ = extract_subregion(stack_path, (x, y, z), size=patch_size)
                filtered = apply_gaussian_smoothing(patch, blur_sigma)
                cx, cy, updated_patch_size, patch = detect_blob_center(filtered)

                if updated_patch_size != patch_size:
                    print(f"Patch size has been updated to {updated_patch_size}. Scaling coordinates appropriately")
                    shrink_px = (patch_size - updated_patch_size) // 2
                    x_min += shrink_px
                    y_min += shrink_px

                # matplotlib_diagnostics(patch, cx, cy)

                refined_x = x_min + cx
                refined_y = y_min + cy

                new_row = row.copy()
                new_row["x"] = refined_x
                new_row["y"] = refined_y
                assert "object_id" in new_row
                assert "contour_id" in new_row
                refined_rows.append(new_row)

                print(f"Original: x={x:.2f}, y={y:.2f}, z={z}")
                print(f"Patch origin: x_min={x_min}, y_min={y_min}")
                print(f"Local center: cx={cx:.2f}, cy={cy:.2f}")
                print(f"Refined global: x={refined_x:.2f}, y={refined_y:.2f}, z={z}")

            except Exception as e:
                print(f"⚠️ Warning: Skipping point in contour {contour_id} due to error: {e}")
                refined_rows.append(row)

    # Save to new file
    refined_df = pd.DataFrame(refined_rows)
    base, ext = os.path.splitext(fid_path)
    backup_fid_path = base + backup_suffix + ext
    os.rename(fid_path, backup_fid_path)
    output_path = base + ext
    imodmodel.write(refined_df, output_path, type=ContourType.OPEN)
    print(f"✅ Saved refined model to: {output_path}")

if __name__ == "__main__":
    for fid_path in glob.glob("*/*.fid", recursive=True):
        base, _ = os.path.splitext(fid_path)
        stack_path = base + "_preali.mrc"
        if fid_path.endswith(f"{BACKUP_SUFFIX}.fid"):
            continue  # Skip backup files
        base, _ = os.path.splitext(fid_path)
        stack_path = base + "_preali.mrc"

        refine_and_save_fiducials(
            fid_path=fid_path,
            stack_path=stack_path,
            backup_suffix=BACKUP_SUFFIX,
            patch_size=PATCH_SIZE,
            blur_sigma=BLUR_SIGMA,
        )
