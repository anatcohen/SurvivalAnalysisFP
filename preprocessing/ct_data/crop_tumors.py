import os
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.draw import polygon
from scipy.ndimage import binary_dilation, zoom


def process_patient(RTSTRUCT_loc, CT_loc):
    """
    Process a patient's CT and RTSTRUCT data to extract, crop, mask, and downsample the tumor region.

    Args:
        RTSTRUCT_loc (str): Path to the RTSTRUCT DICOM file.
        CT_loc (str): Path to the CT scan directory.

    Returns:
        np.ndarray: Downsampled tumor region of shape (64, 64, 32).
    """
    # Load RTSTRUCT
    rtstruct = pydicom.dcmread(RTSTRUCT_loc)

    # Find GTV-1 ROI number
    roi_number = None
    for item in rtstruct.StructureSetROISequence:
        if item.ROIName == "GTV-1":
            roi_number = item.ROINumber
            break
    if roi_number is None:
        raise ValueError("GTV-1 not found in RTSTRUCT")

    # Extract contour data for GTV-1
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
    ct_array = sitk.GetArrayFromImage(ct_image)  # Shape: (z, y, x)
    spacing = np.array(ct_image.GetSpacing())  # (x, y, z)
    origin = np.array(ct_image.GetOrigin())  # (x, y, z)

    # Clip CT values to typical Hounsfield unit range
    ct_array = np.clip(ct_array, -1000, 400)
    ct_array = (ct_array + 1000) / 1400

    # Collect all contour points in physical coordinates
    points = []
    for contour in contour_data:
        contour_points = np.array(contour).reshape(-1, 3)  # (x, y, z)
        points.append(contour_points)
    points = np.vstack(points)

    # Compute tumor centroid in physical coordinates
    centroid = np.mean(points, axis=0)  # (x, y, z)

    # Convert centroid to voxel indices (i, j, k) corresponding to (x, y, z)
    center_voxel = ((centroid - origin) / spacing).round().astype(int)
    center_i, center_j, center_k = center_voxel[0], center_voxel[1], center_voxel[2]

    # Define crop size in voxels (z, y, x)
    crop_size = (64, 128, 128)  # Target: 128x128x64
    half_crop = (32, 64, 64)  # Half sizes for centering

    # Calculate crop indices, ensuring they stay within CT boundaries
    start_k = max(0, center_k - half_crop[0])
    end_k = min(ct_array.shape[0], center_k + half_crop[0])
    start_j = max(0, center_j - half_crop[1])
    end_j = min(ct_array.shape[1], center_j + half_crop[1])
    start_i = max(0, center_i - half_crop[2])
    end_i = min(ct_array.shape[2], center_i + half_crop[2])

    # Extract cropped CT region
    cropped_ct = ct_array[start_k:end_k, start_j:end_j, start_i:end_i]

    # Calculate padding to reach (64, 128, 128)
    pad_z_before = half_crop[0] - (center_k - start_k) if center_k - half_crop[0] < 0 else 0
    pad_z_after = half_crop[0] - (end_k - center_k) if center_k + half_crop[0] > ct_array.shape[0] else 0
    pad_y_before = half_crop[1] - (center_j - start_j) if center_j - half_crop[1] < 0 else 0
    pad_y_after = half_crop[1] - (end_j - center_j) if center_j + half_crop[1] > ct_array.shape[1] else 0
    pad_x_before = half_crop[2] - (center_i - start_i) if center_i - half_crop[2] < 0 else 0
    pad_x_after = half_crop[2] - (end_i - center_i) if center_i + half_crop[2] > ct_array.shape[2] else 0

    # Pad the cropped CT to the desired size
    pad_width = ((pad_z_before, pad_z_after), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after))
    padded_ct = np.pad(cropped_ct, pad_width, mode='constant', constant_values=0)

    target_size = (32, 64, 64)
    zoom_factors = (target_size[0] / padded_ct.shape[0],
                    target_size[1] / padded_ct.shape[1],
                    target_size[2] / padded_ct.shape[2])
    downsampled_ct = zoom(padded_ct, zoom_factors, order=1)

    return downsampled_ct


# Main processing loop
data_dir = "../../data"
df = pd.read_csv(os.path.join(data_dir, "CT_RTSTRUCT_locations.csv"))
output_dir = os.path.join(data_dir, "processed_data")
os.makedirs(output_dir, exist_ok=True)

for ind in df.index:
    subject_id = df['Subject ID'][ind]
    RTSTRUCT_loc = os.path.join(data_dir, df['RTSTRUCT Location'][ind], '1-1.dcm')
    CT_loc = os.path.join(data_dir, df['CT Location'][ind])
    try:
        print(f"Processing Subject ID: {subject_id}")
        downsampled_ct = process_patient(RTSTRUCT_loc, CT_loc)
        # Save the downsampled array
        np.save(os.path.join(output_dir, f"{subject_id}_downsampled.npy"), downsampled_ct)
        print(f"Saved processed data for {subject_id}")
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")