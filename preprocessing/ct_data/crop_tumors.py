import os
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.draw import polygon
from scipy.ndimage import binary_dilation, zoom
import warnings

from config.paths import DATA_DIR

warnings.filterwarnings('ignore')


def create_mask_from_contours(contour_data, ct_shape, spacing, origin):
    """
    Create a binary mask from contour data.

    Args:
        contour_data: List of contour coordinates
        ct_shape: Shape of CT array (z, y, x)
        spacing: Voxel spacing (x, y, z)
        origin: Image origin (x, y, z)

    Returns:
        Binary mask array of same shape as CT
    """
    mask = np.zeros(ct_shape, dtype=bool)

    for contour in contour_data:
        # Reshape contour points
        points = np.array(contour).reshape(-1, 3)  # (n_points, 3) with columns (x, y, z)

        # Get z coordinate (should be same for all points in a contour)
        z_coord = points[0, 2]

        # Convert z coordinate to slice index
        z_idx = int(round((z_coord - origin[2]) / spacing[2]))

        if 0 <= z_idx < ct_shape[0]:
            # Convert x, y coordinates to pixel indices
            x_indices = ((points[:, 0] - origin[0]) / spacing[0]).astype(int)
            y_indices = ((points[:, 1] - origin[1]) / spacing[1]).astype(int)

            # Fill polygon on this slice
            rr, cc = polygon(y_indices, x_indices, shape=(ct_shape[1], ct_shape[2]))
            mask[z_idx, rr, cc] = True

    return mask


def resample_volume(volume, original_spacing, target_spacing, is_mask=False):
    """
    Resample volume to target spacing.

    Args:
        volume: Input volume (z, y, x)
        original_spacing: Original spacing (x, y, z)
        target_spacing: Target spacing (x, y, z)
        is_mask: Whether this is a binary mask (uses nearest neighbor)

    Returns:
        Resampled volume
    """
    # Calculate zoom factors
    # Note: SimpleITK uses (x, y, z) but numpy uses (z, y, x)
    zoom_factors = [
        original_spacing[2] / target_spacing[2],  # z
        original_spacing[1] / target_spacing[1],  # y
        original_spacing[0] / target_spacing[0]  # x
    ]

    # Resample
    order = 0 if is_mask else 1  # Nearest neighbor for masks, linear for CT
    resampled = zoom(volume, zoom_factors, order=order)

    return resampled


def process_patient_physical(RTSTRUCT_loc, CT_loc, box_size_mm=(120, 120, 120),
                             output_size_voxels=(64, 64, 64), target_spacing=(1.0, 1.0, 1.0)):
    """
    Process a patient's CT and RTSTRUCT data with physically-based preprocessing.

    Args:
        RTSTRUCT_loc: Path to RTSTRUCT file
        CT_loc: Path to CT directory
        box_size_mm: Physical size of crop box in mm (x, y, z)
        output_size_voxels: Output size in voxels (x, y, z)
        target_spacing: Target voxel spacing in mm (x, y, z)

    Returns:
        Dict with processed CT, mask, and metadata
    """
    # Load RTSTRUCT
    rtstruct = pydicom.dcmread(RTSTRUCT_loc)

    # Find GTV-1
    roi_number = None
    for item in rtstruct.StructureSetROISequence:
        if item.ROIName == "GTV-1":
            roi_number = item.ROINumber
            break

    if roi_number is None:
        raise ValueError("GTV-1 not found in RTSTRUCT")

    # Extract contours
    contour_data = []
    for contour in rtstruct.ROIContourSequence:
        if contour.ReferencedROINumber == roi_number:
            for contour_seq in contour.ContourSequence:
                coords = contour_seq.ContourData
                contour_data.append(coords)

    if not contour_data:
        raise ValueError("No contours found for GTV-1")

    # Load CT
    reader = sitk.ImageSeriesReader()
    series = reader.GetGDCMSeriesFileNames(CT_loc)
    reader.SetFileNames(series)
    ct_image = reader.Execute()
    ct_array = sitk.GetArrayFromImage(ct_image)  # (z, y, x)
    spacing = np.array(ct_image.GetSpacing())  # (x, y, z)
    origin = np.array(ct_image.GetOrigin())  # (x, y, z)

    # Create tumor mask
    mask = create_mask_from_contours(contour_data, ct_array.shape, spacing, origin)

    # Apply slight dilation to ensure we capture tumor boundaries
    mask = binary_dilation(mask, iterations=4)

    # Find tumor bounding box in physical coordinates
    z_indices, y_indices, x_indices = np.where(mask)

    if len(z_indices) == 0:
        raise ValueError("Empty mask - no tumor voxels found")

    # Convert to physical coordinates
    x_coords = origin[0] + x_indices * spacing[0]
    y_coords = origin[1] + y_indices * spacing[1]
    z_coords = origin[2] + z_indices * spacing[2]

    # Find tumor center in physical space
    center_mm = np.array([
        np.mean(x_coords),
        np.mean(y_coords),
        np.mean(z_coords)
    ])

    # Calculate tumor extent for info
    tumor_size_mm = np.array([
        np.max(x_coords) - np.min(x_coords),
        np.max(y_coords) - np.min(y_coords),
        np.max(z_coords) - np.min(z_coords)
    ])

    print(f"  Tumor size: {tumor_size_mm[0]:.1f} × {tumor_size_mm[1]:.1f} × {tumor_size_mm[2]:.1f} mm")
    print(f"  Box size: {box_size_mm[0]} × {box_size_mm[1]} × {box_size_mm[2]} mm")

    # Define crop region in physical space
    crop_start_mm = center_mm - np.array(box_size_mm) / 2
    crop_end_mm = center_mm + np.array(box_size_mm) / 2

    # Convert to voxel indices
    crop_start_voxel = np.floor((crop_start_mm - origin) / spacing).astype(int)
    crop_end_voxel = np.ceil((crop_end_mm - origin) / spacing).astype(int)

    # Ensure within bounds
    crop_start_voxel = np.maximum(crop_start_voxel, 0)
    crop_end_voxel = np.minimum(crop_end_voxel, [ct_array.shape[2], ct_array.shape[1], ct_array.shape[0]])

    # Extract crop (converting from x,y,z to z,y,x for numpy)
    cropped_ct = ct_array[
                 crop_start_voxel[2]:crop_end_voxel[2],
                 crop_start_voxel[1]:crop_end_voxel[1],
                 crop_start_voxel[0]:crop_end_voxel[0]
                 ]

    cropped_mask = mask[
                   crop_start_voxel[2]:crop_end_voxel[2],
                   crop_start_voxel[1]:crop_end_voxel[1],
                   crop_start_voxel[0]:crop_end_voxel[0]
                   ]

    # Resample to target spacing
    resampled_ct = resample_volume(cropped_ct, spacing, target_spacing, is_mask=False)
    resampled_mask = resample_volume(cropped_mask, spacing, target_spacing, is_mask=True)

    # Resize to target voxel dimensions
    # Convert output_size from (x,y,z) to (z,y,x) for numpy
    target_shape = (output_size_voxels[2], output_size_voxels[1], output_size_voxels[0])

    zoom_factors = [
        target_shape[0] / resampled_ct.shape[0],
        target_shape[1] / resampled_ct.shape[1],
        target_shape[2] / resampled_ct.shape[2]
    ]

    final_ct = zoom(resampled_ct, zoom_factors, order=1)
    final_mask = zoom(resampled_mask, zoom_factors, order=0) > 0.5

    final_ct = np.clip(final_ct, -1000, 400)

    # Option 1: Simple normalization to [0, 1]
    final_ct = (final_ct + 1000) / 1400

    return {
        'ct': final_ct.astype(np.float32),
        'mask': final_mask.astype(np.uint8),
        'tumor_size_mm': tumor_size_mm,
        'tumor_center_mm': center_mm,
        'original_spacing': spacing,
        'output_shape': final_ct.shape
    }


def main():
    """Main processing loop with error handling and progress tracking."""
    df = pd.read_csv(os.path.join(DATA_DIR, "CT_RTSTRUCT_locations.csv"))

    # Create output directory
    output_dir = os.path.join(DATA_DIR, "processed_data_physical")
    os.makedirs(output_dir, exist_ok=True)

    # Processing parameters
    # Based on tumor size analysis, adjust these:
    box_size_mm = (120, 120, 120)  # Adjust based on your histogram analysis
    output_size_voxels = (64, 64, 64)  # CNN input size
    target_spacing = (1.0, 1.0, 1.0)  # 1mm isotropic

    # Track statistics
    successful = 0
    failed = []
    tumor_sizes = []

    print(f"Processing {len(df)} patients...")
    print(f"Parameters:")
    print(f"  Box size: {box_size_mm} mm")
    print(f"  Output size: {output_size_voxels} voxels")
    print(f"  Target spacing: {target_spacing} mm")
    print("-" * 60)

    for idx in df.index:
        subject_id = df['Subject ID'][idx]
        RTSTRUCT_loc = os.path.join(DATA_DIR, df['RTSTRUCT Location'][idx], '1-1.dcm')
        CT_loc = os.path.join(DATA_DIR, df['CT Location'][idx])

        try:
            print(f"\n[{idx + 1}/{len(df)}] Processing {subject_id}...")

            # Process patient
            result = process_patient_physical(
                RTSTRUCT_loc, CT_loc,
                box_size_mm=box_size_mm,
                output_size_voxels=output_size_voxels,
                target_spacing=target_spacing
            )

            # Save processed data
            np.savez_compressed(
                os.path.join(output_dir, f"{subject_id}_processed.npz"),
                ct=result['ct'],
                mask=result['mask'],
                tumor_size_mm=result['tumor_size_mm'],
                tumor_center_mm=result['tumor_center_mm']
            )

            tumor_sizes.append(result['tumor_size_mm'])
            successful += 1
            print(f"  ✓ Saved successfully")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed.append((subject_id, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successful: {successful}/{len(df)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed cases:")
        for sid, error in failed[:10]:  # Show first 10
            print(f"  {sid}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    # Check if any tumors were too large for the box
    if tumor_sizes:
        tumor_sizes = np.array(tumor_sizes)
        max_sizes = np.max(tumor_sizes, axis=0)
        print(f"\nMaximum tumor size encountered: {max_sizes[0]:.1f} × {max_sizes[1]:.1f} × {max_sizes[2]:.1f} mm")

        oversized = np.any(tumor_sizes > box_size_mm, axis=1).sum()
        if oversized > 0:
            print(f"WARNING: {oversized} tumors exceeded box size in at least one dimension!")
            print("Consider increasing box_size_mm or using adaptive sizing")


if __name__ == "__main__":
    main()