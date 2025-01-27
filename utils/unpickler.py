import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import re
from datetime import datetime
from glob import glob
from pathlib import Path
import sys

# Add parent directory to Python path to import seizure_event_tagger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seizure_event_tagger import tag_embeddings_with_seizures

def parse_timestamp(timestamp_str):
    """Parse timestamp from filename format MMDDYYYY_HHMMSSCC."""
    date = datetime.strptime(timestamp_str[:8], '%m%d%Y')
    hour = int(timestamp_str[9:11])
    minute = int(timestamp_str[11:13])
    seconds = int(timestamp_str[13:15])
    centiseconds = int(timestamp_str[15:])
    # Convert centiseconds to microseconds (multiply by 10000 to get microseconds)
    microseconds = centiseconds * 10000
    
    return pd.Timestamp(date.year, date.month, date.day, hour, minute, seconds, microseconds)

def get_patient_files(base_dir, animal, window_length, stride_length, data_type):
    """Get all pickle files for all patients in the specified directory."""
    pattern = os.path.join(base_dir, 'source_pickles', animal, f'Epoch*',
                          f'{window_length}SecondWindow_{stride_length}SecondStride',
                          data_type, '*.pkl')
    files = glob(pattern)
    
    # Group files by patient ID
    patient_files = {}
    for file in files:
        filename = os.path.basename(file)
        match = re.match(r'(Epat\d+)_.*\.pkl', filename)
        if match:
            patient_id = match.group(1)
            if patient_id not in patient_files:
                patient_files[patient_id] = []
            patient_files[patient_id].append(file)
    
    return patient_files

def setup_output_directory(animal, patient_id):
    """Create output directory structure for the patient."""
    output_dir = os.path.join('output', animal, patient_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def process_patient_files(patient_files, animal, patient_id, window_length, stride_length, data_type):
    """Process all files for a single patient."""
    print(f"\n=== Processing {patient_id} ===")
    output_dir = setup_output_directory(animal, patient_id)
    
    # Sort files by timestamp to maintain order
    sorted_files = sorted(patient_files)
    
    # Process embeddings
    all_embeddings = []
    start_times = []
    stop_times = []
    
    for file_path in sorted_files:
        filename = os.path.basename(file_path)
        # Extract timestamps from filename
        match = re.search(r'(\d{8}_\d{8})_to_(\d{8}_\d{8})', filename)
        if not match:
            print(f"Skipping file with invalid format: {filename}")
            continue
            
        start_time = parse_timestamp(match.group(1))
        stop_time = parse_timestamp(match.group(2))
        
        # Load embedding
        with open(file_path, 'rb') as f:
            embedding = pickle.load(f)
        
        all_embeddings.append(embedding)
        start_times.append(start_time)
        stop_times.append(stop_time)
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings)
    
    # Create output data structure
    output_data = {
        'patient_id': patient_id,
        'patient_embeddings': embeddings_array,
        'start_times': start_times,
        'stop_times': stop_times,
        'file_indices': list(range(len(start_times))),
        'window_indices': list(range(len(start_times))),
        'original_shape': embeddings_array.shape,
        'sleep_labels': None
    }
    
    # Tag seizures
    try:
        print(f"\nTagging seizures for {patient_id}...")
        seizure_labels = tag_embeddings_with_seizures(output_data, patient_id)
        output_data['seizure_labels'] = seizure_labels
        
        # Verify seizure tagging
        if seizure_labels is not None:
            print("\nVerification of seizure tagging:")
            print(f"Shape of seizure labels: {seizure_labels.shape}")
            labeled_windows = np.sum(seizure_labels == 1)
            print(f"Total windows with seizures: {labeled_windows}")
            if labeled_windows > 0:
                print("\nSample of seizure windows:")
                seizure_indices = np.where(seizure_labels == 1)[0][:5]  # Show first 5 seizure windows
                for idx in seizure_indices:
                    print(f"Window {idx}:")
                    print(f"  Start: {start_times[idx]}")
                    print(f"  Stop: {stop_times[idx]}")
        
        print(f"\nSuccessfully tagged seizures for {patient_id}")
    except Exception as e:
        print(f"Warning: Failed to tag seizures for {patient_id}: {e}")
        output_data['seizure_labels'] = None
    
    # Save processed data
    output_filename = f'embeddings_{patient_id}_W{window_length}_S{stride_length}_{data_type}.pkl'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Processing complete. File saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Process brain state embeddings from individual window files.')
    parser.add_argument('--animal', type=str, required=True,
                      help='Animal name (e.g., rhesusmonkey)')
    parser.add_argument('--data_type', type=str, required=True, choices=['train', 'valfinetune', 'valunseen'],
                      help='Data type folder to process')
    parser.add_argument('--window_length', type=int, default=60,
                      help='Window length in seconds (default: 60)')
    parser.add_argument('--stride_length', type=int, default=30,
                      help='Stride length in seconds (default: 30)')
    
    args = parser.parse_args()
    
    try:
        # Get all patient files
        patient_files = get_patient_files('.', args.animal, args.window_length, 
                                        args.stride_length, args.data_type)
        
        if not patient_files:
            print(f"No patient files found for animal {args.animal} with the specified parameters")
            return
        
        print(f"\nFound {len(patient_files)} patients to process")
        
        # Process each patient
        processed_files = []
        for patient_id, files in patient_files.items():
            try:
                output_path = process_patient_files(files, args.animal, patient_id,
                                                 args.window_length, args.stride_length,
                                                 args.data_type)
                processed_files.append(output_path)
            except Exception as e:
                print(f"Error processing {patient_id}: {e}")
                continue
        
        print(f"\nProcessing complete. Processed {len(processed_files)} patients successfully.")
        
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)

if __name__ == "__main__":
    main()