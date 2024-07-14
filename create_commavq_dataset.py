from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm

# Load all splits of the dataset
dataset_name = "commaai/commavq"
dataset = load_dataset(dataset_name, split='0', trust_remote_code=True)

# Function to process and tokenize the dataset
def process_and_tokenize(dataset):
    token_sequences = []
    for data in tqdm(dataset):
        frames = np.load(data['path']).astype(np.int16).reshape(-1, 128)
        for frame in frames: token_sequences.append(list(frame))
    return token_sequences

tokenized_sequences = process_and_tokenize(dataset)

# Create the combined dataset
batch_size = 100

# Initialize an empty dataset
combined_tokenized_dataset = Dataset.from_dict({"input_ids": []})

# Iterate through the tokenized sequences in batches of 100
for i in tqdm(range(0, len(tokenized_sequences), batch_size)):
    batch = tokenized_sequences[i:i+batch_size]
    temp_dataset = Dataset.from_dict({"input_ids": batch})
    combined_tokenized_dataset = concatenate_datasets([temp_dataset, combined_tokenized_dataset])

# Save the combined dataset to disk
combined_tokenized_dataset.save_to_disk("commavq-dataset-split-0")
print("Combined dataset saved successfully.")

