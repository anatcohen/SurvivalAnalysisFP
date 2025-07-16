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
    Returns dimensions in both pixels and mm: (dimensions_pixels, dimensions_mm, spacing)
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

        return dimensions_pixels, dimensions_mm, spacing

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
        result = extract_tumor_dimensions(rtstruct_path, ct_path)

        if result is not None:
            dimensions_pixels, dimensions_mm, spacing = result
            results.append({
                'Subject_ID': subject_id,
                'X_pixels': dimensions_pixels[0],
                'Y_pixels': dimensions_pixels[1],
                'Z_pixels': dimensions_pixels[2],
                'X_mm': dimensions_mm[0],
                'Y_mm': dimensions_mm[1],
                'Z_mm': dimensions_mm[2],
                'Spacing_X': spacing[0],
                'Spacing_Y': spacing[1],
                'Spacing_Z': spacing[2]
            })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = os.path.join(data_dir, "tumor_dimensions_analysis.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved dimensions to {output_path}")

    return results_df


def create_histograms(results_df, data_dir):
    """
    Create histograms for tumor dimensions along each axis (both pixels and mm)
    """
    # Create figure with 2 rows: top for pixels, bottom for mm
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Tumor Dimensions Distribution', fontsize=16)

    # Color scheme
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    axes_labels = ['X', 'Y', 'Z']

    # Process pixel dimensions (top row)
    for i, axis in enumerate(axes_labels):
        data_pixels = results_df[f'{axis}_pixels']
        ax = axes[0, i]

        ax.hist(data_pixels, bins=30, color=colors[i], edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'{axis} Dimension (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{axis}-axis (Pixels)')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = data_pixels.mean()
        std_val = data_pixels.std()
        max_val = data_pixels.max()
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(max_val, color='green', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.1f}')
        ax.legend()

    # Process mm dimensions (bottom row)
    for i, axis in enumerate(axes_labels):
        data_mm = results_df[f'{axis}_mm']
        ax = axes[1, i]

        ax.hist(data_mm, bins=30, color=colors[i], edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'{axis} Dimension (mm)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{axis}-axis (Millimeters)')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = data_mm.mean()
        std_val = data_mm.std()
        max_val = data_mm.max()
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(max_val, color='green', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.1f}')
        ax.legend()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(data_dir, "tumor_dimensions_histograms.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved histograms to {output_path}")

    plt.show()

    # Print summary statistics for both pixels and mm
    print("\nSummary Statistics:")
    print("=" * 70)
    print("PIXEL DIMENSIONS:")
    print("-" * 70)
    for axis in axes_labels:
        data = results_df[f'{axis}_pixels']
        print(f"{axis}-axis: Mean={data.mean():.1f}, Std={data.std():.1f}, Max={data.max():.1f} pixels")

    print("\nMILLIMETER DIMENSIONS:")
    print("-" * 70)
    for axis in axes_labels:
        data = results_df[f'{axis}_mm']
        print(f"{axis}-axis: Mean={data.mean():.1f}, Std={data.std():.1f}, Max={data.max():.1f} mm")

    # Print spacing statistics
    print("\nVOXEL SPACING STATISTICS:")
    print("-" * 70)
    for axis in axes_labels:
        spacing_data = results_df[f'Spacing_{axis}']
        print(f"{axis}-axis spacing: Mean={spacing_data.mean():.3f}, Std={spacing_data.std():.3f} mm/pixel")

    # Calculate suggested box sizes
    print("\n" + "=" * 70)
    print("SUGGESTED BOX SIZES FOR CNN INPUT:")
    print("=" * 70)

    # Pixel-based recommendations
    print("\nPIXEL-BASED RECOMMENDATIONS:")
    print("-" * 70)

    # Get max dimensions for each approach
    x_pixels = results_df['X_pixels']
    y_pixels = results_df['Y_pixels']
    z_pixels = results_df['Z_pixels']

    # Option 1: Maximum dimensions
    box_size_max = int(np.ceil(max(x_pixels.max(), y_pixels.max(), z_pixels.max())))
    print(f"1. Maximum dimension approach: {box_size_max}³ pixels")

    # Option 2: Mean + 2*std
    box_size_95 = int(np.ceil(max(
        x_pixels.mean() + 2 * x_pixels.std(),
        y_pixels.mean() + 2 * y_pixels.std(),
        z_pixels.mean() + 2 * z_pixels.std()
    )))
    print(f"2. Mean + 2*std approach (covers ~95%): {box_size_95}³ pixels")

    # Option 3: 99th percentile
    percentile_99 = int(np.ceil(max(
        x_pixels.quantile(0.99),
        y_pixels.quantile(0.99),
        z_pixels.quantile(0.99)
    )))
    print(f"3. 99th percentile approach: {percentile_99}³ pixels")

    # Option 4: CNN-friendly sizes
    common_sizes = [32, 64, 96, 128, 160, 192, 224, 256]
    recommended_size = min([s for s in common_sizes if s >= box_size_95])
    print(f"4. CNN-friendly size (nearest power-friendly): {recommended_size}³ pixels")

    # MM-based recommendations
    print("\nMILLIMETER-BASED RECOMMENDATIONS:")
    print("-" * 70)

    x_mm = results_df['X_mm']
    y_mm = results_df['Y_mm']
    z_mm = results_df['Z_mm']

    # Physical dimensions
    phys_max = max(x_mm.max(), y_mm.max(), z_mm.max())
    phys_95 = max(
        x_mm.mean() + 2 * x_mm.std(),
        y_mm.mean() + 2 * y_mm.std(),
        z_mm.mean() + 2 * z_mm.std()
    )
    phys_99 = max(x_mm.quantile(0.99), y_mm.quantile(0.99), z_mm.quantile(0.99))

    print(f"1. Maximum physical dimension: {phys_max:.1f} mm")
    print(f"2. Mean + 2*std physical dimension: {phys_95:.1f} mm")
    print(f"3. 99th percentile physical dimension: {phys_99:.1f} mm")

    # Calculate required pixels for physical coverage at different resolutions
    print("\nREQUIRED PIXELS FOR PHYSICAL COVERAGE:")
    print("-" * 70)
    common_resolutions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for res in common_resolutions:
        pixels_needed = int(np.ceil(phys_95 / res))
        print(f"At {res} mm/pixel resolution: {pixels_needed}³ pixels needed")

    return box_size_max, box_size_95, percentile_99, recommended_size


def create_aspect_ratio_analysis(results_df, data_dir):
    """
    Create additional analysis for aspect ratios and anisotropy
    """
    # Calculate aspect ratios for both pixels and mm
    results_df['XY_ratio_pixels'] = results_df['X_pixels'] / results_df['Y_pixels']
    results_df['XZ_ratio_pixels'] = results_df['X_pixels'] / results_df['Z_pixels']
    results_df['YZ_ratio_pixels'] = results_df['Y_pixels'] / results_df['Z_pixels']

    results_df['XY_ratio_mm'] = results_df['X_mm'] / results_df['Y_mm']
    results_df['XZ_ratio_mm'] = results_df['X_mm'] / results_df['Z_mm']
    results_df['YZ_ratio_mm'] = results_df['Y_mm'] / results_df['Z_mm']

    # Create aspect ratio visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Tumor Aspect Ratios', fontsize=16)

    # Pixel-based aspect ratios
    ax = axes[0]
    ratios_pixels = [results_df['XY_ratio_pixels'], results_df['XZ_ratio_pixels'], results_df['YZ_ratio_pixels']]
    labels = ['X/Y', 'X/Z', 'Y/Z']
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    bp1 = ax.boxplot(ratios_pixels, labels=labels, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Aspect Ratio')
    ax.set_title('Pixel-based Aspect Ratios')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Isotropic (1:1)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # MM-based aspect ratios
    ax = axes[1]
    ratios_mm = [results_df['XY_ratio_mm'], results_df['XZ_ratio_mm'], results_df['YZ_ratio_mm']]

    bp2 = ax.boxplot(ratios_mm, labels=labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Aspect Ratio')
    ax.set_title('MM-based Aspect Ratios')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Isotropic (1:1)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(data_dir, "tumor_aspect_ratios.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved aspect ratio analysis to {output_path}")

    plt.show()

    # Print aspect ratio statistics
    print("\n" + "=" * 70)
    print("ASPECT RATIO ANALYSIS:")
    print("=" * 70)

    print("\nPIXEL-BASED ASPECT RATIOS:")
    print("-" * 70)
    for ratio_name, ratio_col in zip(['X/Y', 'X/Z', 'Y/Z'],
                                     ['XY_ratio_pixels', 'XZ_ratio_pixels', 'YZ_ratio_pixels']):
        data = results_df[ratio_col]
        print(f"{ratio_name}: Mean={data.mean():.2f}, Std={data.std():.2f}, "
              f"Min={data.min():.2f}, Max={data.max():.2f}")

    print("\nMM-BASED ASPECT RATIOS:")
    print("-" * 70)
    for ratio_name, ratio_col in zip(['X/Y', 'X/Z', 'Y/Z'],
                                     ['XY_ratio_mm', 'XZ_ratio_mm', 'YZ_ratio_mm']):
        data = results_df[ratio_col]
        print(f"{ratio_name}: Mean={data.mean():.2f}, Std={data.std():.2f}, "
              f"Min={data.min():.2f}, Max={data.max():.2f}")

    # Check anisotropy
    max_ratio_pixels = max(
        results_df['XY_ratio_pixels'].max(),
        results_df['XZ_ratio_pixels'].max(),
        results_df['YZ_ratio_pixels'].max(),
        1 / results_df['XY_ratio_pixels'].min(),
        1 / results_df['XZ_ratio_pixels'].min(),
        1 / results_df['YZ_ratio_pixels'].min()
    )

    max_ratio_mm = max(
        results_df['XY_ratio_mm'].max(),
        results_df['XZ_ratio_mm'].max(),
        results_df['YZ_ratio_mm'].max(),
        1 / results_df['XY_ratio_mm'].min(),
        1 / results_df['XZ_ratio_mm'].min(),
        1 / results_df['YZ_ratio_mm'].min()
    )

    print("\nANISOTROPY ASSESSMENT:")
    print("-" * 70)
    print(f"Maximum aspect ratio (pixels): {max_ratio_pixels:.1f}")
    print(f"Maximum aspect ratio (mm): {max_ratio_mm:.1f}")

    if max_ratio_pixels > 3 or max_ratio_mm > 3:
        print("\nWarning: High anisotropy detected!")
        print("Consider using anisotropic box sizes for better efficiency")

        # Suggest anisotropic box sizes
        print("\nSUGGESTED ANISOTROPIC BOX SIZES:")
        print("-" * 70)

        # For pixels
        x_size = int(np.ceil(results_df['X_pixels'].mean() + 2 * results_df['X_pixels'].std()))
        y_size = int(np.ceil(results_df['Y_pixels'].mean() + 2 * results_df['Y_pixels'].std()))
        z_size = int(np.ceil(results_df['Z_pixels'].mean() + 2 * results_df['Z_pixels'].std()))

        print(f"Pixel-based (mean + 2*std): {x_size} × {y_size} × {z_size} pixels")

        # For mm
        x_size_mm = results_df['X_mm'].mean() + 2 * results_df['X_mm'].std()
        y_size_mm = results_df['Y_mm'].mean() + 2 * results_df['Y_mm'].std()
        z_size_mm = results_df['Z_mm'].mean() + 2 * results_df['Z_mm'].std()

        print(f"MM-based (mean + 2*std): {x_size_mm:.1f} × {y_size_mm:.1f} × {z_size_mm:.1f} mm")
    else:
        print("\nTumors are relatively isotropic - cubic boxes are appropriate")


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

        # Additional aspect ratio analysis
        create_aspect_ratio_analysis(results_df, data_dir)

        # Save detailed statistics
        stats_path = os.path.join(data_dir, "tumor_dimension_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("TUMOR DIMENSION ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total patients analyzed: {len(results_df)}\n\n")

            f.write("DIMENSION STATISTICS (PIXELS):\n")
            f.write("-" * 70 + "\n")
            for axis in ['X', 'Y', 'Z']:
                data = results_df[f'{axis}_pixels']
                f.write(f"{axis}-axis: Mean={data.mean():.1f}, Std={data.std():.1f}, "
                        f"Min={data.min():.1f}, Max={data.max():.1f}\n")

            f.write("\nDIMENSION STATISTICS (MM):\n")
            f.write("-" * 70 + "\n")
            for axis in ['X', 'Y', 'Z']:
                data = results_df[f'{axis}_mm']
                f.write(f"{axis}-axis: Mean={data.mean():.1f}, Std={data.std():.1f}, "
                        f"Min={data.min():.1f}, Max={data.max():.1f}\n")

            f.write("\nRECOMMENDED BOX SIZE: {}³ pixels\n".format(box_sizes[3]))

        print(f"\nDetailed statistics saved to {stats_path}")

    else:
        print("No valid tumor dimensions found!")


if __name__ == "__main__":
    main()