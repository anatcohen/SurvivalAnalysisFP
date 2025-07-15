import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm


def extract_tumor_dimensions(rtstruct_path, ct_path):
    """
    Extract the bounding box dimensions of GTV-1 from RTSTRUCT file
    Returns dimensions in pixels (x_length, y_length, z_length)
    """
    try:
        # Load RTSTRUCT
        rtstruct = pydicom.dcmread(rtstruct_path)

        # Load CT to get spacing information
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(ct_path)
        reader.SetFileNames(series)
        ct_image = reader.Execute()
        spacing = np.array(ct_image.GetSpacing())
        origin = np.array(ct_image.GetOrigin())

        # Find GTV-1 ROI number
        roi_number = None
        for item in rtstruct.StructureSetROISequence:
            if item.ROIName == "GTV-1":
                roi_number = item.ROINumber
                break

        if roi_number is None:
            print(f"GTV-1 not found in {rtstruct_path}")
            return None

        # Extract all contour points
        all_points = []
        for contour in rtstruct.ROIContourSequence:
            if contour.ReferencedROINumber == roi_number:
                for contour_seq in contour.ContourSequence:
                    coords = np.array(contour_seq.ContourData).reshape(-1, 3)
                    all_points.extend(coords)

        if not all_points:
            print(f"No contour points found for GTV-1 in {rtstruct_path}")
            return None

        all_points = np.array(all_points)

        # Find bounding box in physical coordinates
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)

        # Calculate dimensions in mm
        dimensions_mm = max_coords - min_coords

        # Convert to pixels
        dimensions_pixels = dimensions_mm / spacing

        return dimensions_pixels

    except Exception as e:
        print(f"Error processing {rtstruct_path}: {str(e)}")
        return None


def analyze_all_patients(data_dir, csv_file):
    """
    Analyze tumor dimensions for all patients in the CSV file
    """
    # Load the CSV with patient data
    df = pd.read_csv(os.path.join(data_dir, csv_file))

    # Store results
    results = []

    print(f"Processing {len(df)} patients...")

    # Process each patient
    for idx in tqdm(range(len(df))):
        subject_id = df['Subject ID'][idx]

        # Construct full paths
        rtstruct_path = os.path.join(data_dir, df['RTSTRUCT Location'][idx], '1-1.dcm')
        ct_path = os.path.join(data_dir, df['CT Location'][idx])

        # Extract dimensions
        dimensions = extract_tumor_dimensions(rtstruct_path, ct_path)

        if dimensions is not None:
            results.append({
                'Subject_ID': subject_id,
                'X_pixels': dimensions[0],
                'Y_pixels': dimensions[1],
                'Z_pixels': dimensions[2],
            })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Correct the mm calculations (need spacing info)
    # For now, we'll just store pixel dimensions
    results_df = results_df[['Subject_ID', 'X_pixels', 'Y_pixels', 'Z_pixels']]

    # Save to CSV
    output_path = os.path.join(data_dir, "tumor_dimensions_analysis.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved dimensions to {output_path}")

    return results_df


def create_histograms(results_df, data_dir):
    """
    Create histograms for tumor dimensions along each axis
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Tumor Dimensions Distribution (in pixels)', fontsize=16)

    # X-axis histogram
    axes[0].hist(results_df['X_pixels'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('X Dimension (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('X-axis')
    axes[0].grid(True, alpha=0.3)

    # Add statistics
    x_mean = results_df['X_pixels'].mean()
    x_std = results_df['X_pixels'].std()
    x_max = results_df['X_pixels'].max()
    axes[0].axvline(x_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {x_mean:.1f}')
    axes[0].axvline(x_max, color='green', linestyle='dashed', linewidth=2, label=f'Max: {x_max:.1f}')
    axes[0].legend()

    # Y-axis histogram
    axes[1].hist(results_df['Y_pixels'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Y Dimension (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Y-axis')
    axes[1].grid(True, alpha=0.3)

    # Add statistics
    y_mean = results_df['Y_pixels'].mean()
    y_std = results_df['Y_pixels'].std()
    y_max = results_df['Y_pixels'].max()
    axes[1].axvline(y_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {y_mean:.1f}')
    axes[1].axvline(y_max, color='green', linestyle='dashed', linewidth=2, label=f'Max: {y_max:.1f}')
    axes[1].legend()

    # Z-axis histogram
    axes[2].hist(results_df['Z_pixels'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Z Dimension (pixels)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Z-axis')
    axes[2].grid(True, alpha=0.3)

    # Add statistics
    z_mean = results_df['Z_pixels'].mean()
    z_std = results_df['Z_pixels'].std()
    z_max = results_df['Z_pixels'].max()
    axes[2].axvline(z_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {z_mean:.1f}')
    axes[2].axvline(z_max, color='green', linestyle='dashed', linewidth=2, label=f'Max: {z_max:.1f}')
    axes[2].legend()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(data_dir, "tumor_dimensions_histograms.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved histograms to {output_path}")

    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"X-axis: Mean={x_mean:.1f}, Std={x_std:.1f}, Max={x_max:.1f} pixels")
    print(f"Y-axis: Mean={y_mean:.1f}, Std={y_std:.1f}, Max={y_max:.1f} pixels")
    print(f"Z-axis: Mean={z_mean:.1f}, Std={z_std:.1f}, Max={z_max:.1f} pixels")
    print("-" * 50)

    # Suggest box sizes
    print("\nSuggested box sizes for CNN input:")
    print("-" * 50)

    # Option 1: Use maximum dimensions (ensures all tumors fit)
    box_size_max = int(np.ceil(max(x_max, y_max, z_max)))
    print(f"1. Maximum dimension approach: {box_size_max}³ pixels")

    # Option 2: Use mean + 2*std (covers ~95% of tumors)
    box_size_95 = int(np.ceil(max(
        x_mean + 2 * x_std,
        y_mean + 2 * y_std,
        z_mean + 2 * z_std
    )))
    print(f"2. Mean + 2*std approach: {box_size_95}³ pixels")

    # Option 3: Use percentile approach
    percentile_99 = int(np.ceil(max(
        results_df['X_pixels'].quantile(0.99),
        results_df['Y_pixels'].quantile(0.99),
        results_df['Z_pixels'].quantile(0.99)
    )))
    print(f"3. 99th percentile approach: {percentile_99}³ pixels")

    # Option 4: Common CNN-friendly sizes
    common_sizes = [32, 64, 96, 128, 160, 192, 224, 256]
    recommended_size = min([s for s in common_sizes if s >= box_size_95])
    print(f"4. CNN-friendly size (nearest power-friendly): {recommended_size}³ pixels")

    return box_size_max, box_size_95, percentile_99, recommended_size


def main():
    # Set data directory
    data_dir = "../../data"
    csv_file = "CT_RTSTRUCT_locations.csv"

    # Analyze all patients
    print("Starting tumor dimension analysis...")
    results_df = analyze_all_patients(data_dir, csv_file)

    if len(results_df) > 0:
        print(f"\nSuccessfully analyzed {len(results_df)} patients")

        # Create histograms and get recommendations
        box_sizes = create_histograms(results_df, data_dir)

        # Additional analysis: aspect ratios
        print("\n\nAspect Ratio Analysis:")
        print("-" * 50)
        results_df['XY_ratio'] = results_df['X_pixels'] / results_df['Y_pixels']
        results_df['XZ_ratio'] = results_df['X_pixels'] / results_df['Z_pixels']
        results_df['YZ_ratio'] = results_df['Y_pixels'] / results_df['Z_pixels']

        print(f"X/Y ratio: Mean={results_df['XY_ratio'].mean():.2f}, Std={results_df['XY_ratio'].std():.2f}")
        print(f"X/Z ratio: Mean={results_df['XZ_ratio'].mean():.2f}, Std={results_df['XZ_ratio'].std():.2f}")
        print(f"Y/Z ratio: Mean={results_df['YZ_ratio'].mean():.2f}, Std={results_df['YZ_ratio'].std():.2f}")

        # Check if isotropic box is reasonable
        max_ratio = max(
            results_df['XY_ratio'].max(),
            results_df['XZ_ratio'].max(),
            results_df['YZ_ratio'].max(),
            1 / results_df['XY_ratio'].min(),
            1 / results_df['XZ_ratio'].min(),
            1 / results_df['YZ_ratio'].min()
        )

        if max_ratio > 3:
            print(f"\nWarning: Maximum aspect ratio is {max_ratio:.1f}")
            print("Consider using anisotropic box sizes for better efficiency")
    else:
        print("No valid tumor dimensions found!")


if __name__ == "__main__":
    main()