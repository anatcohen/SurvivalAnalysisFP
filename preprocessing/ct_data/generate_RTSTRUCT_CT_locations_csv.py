import os
import pandas as pd
from config.paths import DATA_DIR

def process_medical_csv(input_file, output_file):
    """
    Process a CSV file containing medical imaging data to verify modality constraints
    and create a reorganized output file.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """

    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {input_file}")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Verify required columns exist
    required_columns = ['Subject ID', 'Modality', 'File Location']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return

    # Group by Subject ID and check modality counts
    print("\nVerifying modality constraints...")
    issues = []

    for subject_id, group in df.groupby('Subject ID'):
        ct_count = len(group[group['Modality'] == 'CT'])
        rtstruct_count = len(group[group['Modality'] == 'RTSTRUCT'])

        if ct_count != 1:
            issues.append(f"Subject {subject_id}: Found {ct_count} CT entries (expected 1)")
        if rtstruct_count != 1:
            issues.append(f"Subject {subject_id}: Found {rtstruct_count} RTSTRUCT entries (expected 1)")

    if issues:
        print("\nWarning: Found the following constraint violations:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All subjects have exactly one CT and one RTSTRUCT entry.")

    # Get unique subject IDs in alphabetical order
    subject_ids = sorted(df['Subject ID'].unique())

    # Initialize lists for the new dataframe
    new_data = {
        'Subject ID': [],
        'RTSTRUCT Location': [],
        'CT Location': []
    }

    # Process each subject
    for subject_id in subject_ids:
        subject_data = df[df['Subject ID'] == subject_id]

        # Get RTSTRUCT location
        rtstruct_data = subject_data[subject_data['Modality'] == 'RTSTRUCT']
        rtstruct_location = rtstruct_data['File Location'].iloc[0] if len(rtstruct_data) > 0 else 'N/A'

        # Get CT location
        ct_data = subject_data[subject_data['Modality'] == 'CT']
        ct_location = ct_data['File Location'].iloc[0] if len(ct_data) > 0 else 'N/A'

        # Add to new data
        new_data['Subject ID'].append(subject_id)
        new_data['RTSTRUCT Location'].append(rtstruct_location)
        new_data['CT Location'].append(ct_location)

    # Create new dataframe
    new_df = pd.DataFrame(new_data)

    # Save to CSV
    try:
        new_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved reorganized data to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return

    return new_df


# Example usage
if __name__ == "__main__":
    input_file = os.path.join(DATA_DIR, 'CT_metadata.csv')
    output_file = os.path.join(DATA_DIR, 'CT_RTSTRUCT_locations.csv')

    # Process the CSV file
    result_df = process_medical_csv(input_file, output_file)

    if result_df is not None:
        pd.set_option('display.width', None)
        print(result_df)
    else:
        print("\nNo data found. Exiting...")
