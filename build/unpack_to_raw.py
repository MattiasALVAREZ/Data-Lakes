import os
import pandas as pd
import boto3
import tempfile  # Import pour gÃ©rer les dossiers temporaires


def unpack_data(input_dir, output_file, bucket_name):
    """
    Unpacks and combines multiple CSV files from subdirectories into a single CSV file
    and uploads it to an S3 bucket.

    Parameters:
    input_dir (str): Path to the directory containing the subfolders `train`, `test`, `dev`.
    output_file (str): Name of the combined output CSV file.
    bucket_name (str): S3 bucket name for uploading the file.
    """
    data_frames = []

    # Iterate through subdirectories
    for subfolder in ["train", "test", "dev"]:
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.exists(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.csv') or 'data-' in file_name:
                    file_path = os.path.join(subfolder_path, file_name)
                    data = pd.read_csv(
                        file_path,
                        names=['sequence', 'family_accession', 'sequence_name', 'aligned_sequence', 'family_id']
                    )
                    data_frames.append(data)
    
    # Combine all data frames into one
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Save to temporary directory
    temp_file_path = os.path.join(tempfile.gettempdir(), output_file)
    combined_data.to_csv(temp_file_path, index=False)

    # Upload to S3
    s3_client = boto3.client('s3', endpoint_url='http://localhost:4566')
    s3_client.upload_file(temp_file_path, bucket_name, output_file)

    print(f"Combined file uploaded to bucket '{bucket_name}' as '{output_file}'.")
    print(f"Temporary file saved at: {temp_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack and combine protein data, then upload to S3")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory containing subfolders")
    parser.add_argument("--bucket_name", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--output_file_name", type=str, required=True, help="Name of the output combined CSV file")
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file_name, args.bucket_name)