import io
import pandas as pd
import boto3
from transformers import AutoTokenizer
import tempfile


def process_to_curated(bucket_staging, bucket_curated, input_file, output_file, model_name):
    # Step 1: Initialize S3 client
    s3 = boto3.client('s3', endpoint_url='http://localhost:4566')

    # Step 2: Download the input file from the staging bucket
    print(f"Downloading {input_file} from bucket {bucket_staging}...")
    response = s3.get_object(Bucket=bucket_staging, Key=input_file)
    data = pd.read_csv(io.BytesIO(response['Body'].read()))

    # Ensure the input file contains a 'sequence' column
    if 'sequence' not in data.columns:
        raise ValueError("Input file must contain a 'sequence' column.")

    # Step 3: Initialize the tokenizer
    print(f"Loading tokenizer for model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 4: Tokenize the sequences
    print("Tokenizing sequences...")
    tokenized_sequences = []
    for seq in data['sequence']:
        tokenized = tokenizer(
            seq,
            truncation=True,
            padding='max_length',
            max_length=1024,
            return_tensors="np"
        )
        tokenized_sequences.append(tokenized['input_ids'][0])

    # Step 5: Create a DataFrame for tokenized sequences
    tokenized_df = pd.DataFrame(tokenized_sequences, columns=[f"token_{i}" for i in range(1024)])

    # Step 6: Merge the tokenized data with the metadata
    metadata = data.drop(columns=['sequence'])
    processed_data = pd.concat([metadata, tokenized_df], axis=1)

    # Step 7: Save the processed data locally
    print("Saving processed data locally...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        processed_file_path = tmp_file.name
        processed_data.to_csv(processed_file_path, index=False)

    # Step 8: Upload the processed file to the curated bucket
    print(f"Uploading processed file to bucket {bucket_curated} as {output_file}...")
    with open(processed_file_path, 'rb') as f:
        s3.upload_fileobj(f, bucket_curated, output_file)

    print("Processing and upload completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process data from staging to curated bucket")
    parser.add_argument("--bucket_staging", type=str, required=True, help="Name of the staging S3 bucket")
    parser.add_argument("--bucket_curated", type=str, required=True, help="Name of the curated S3 bucket")
    parser.add_argument("--input_file", type=str, required=True, help="Name of the input file in the staging bucket")
    parser.add_argument("--output_file", type=str, required=True, help="Name of the output file in the curated bucket")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Tokenizer model name")
    args = parser.parse_args()

    process_to_curated(args.bucket_staging, args.bucket_curated, args.input_file, args.output_file, args.model_name)