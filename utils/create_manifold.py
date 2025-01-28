import numpy as np
from pacmap import PaCMAP
from pacmap import sample_neighbors_pair
import pickle
import hdbscan
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from unpickler import process_patient_files as unpack_single_patient
import re
import glob

def setup_output_directory(animal, patient_id):
    """Create output directory structure for the patient."""
    output_dir = os.path.join('output', animal, f"Epat{patient_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def find_embeddings_file(animal, patient_id, window_length=60, stride_length=30, data_type='train'):
    """Find the embeddings file for a patient. If not found, generate it."""
    # Convert integer to Epat format
    patient_id_str = f"Epat{patient_id}"
    
    output_dir = os.path.join('output', animal, patient_id_str)
    version_file = f'embeddings_{patient_id_str}_{window_length}win{stride_length}str_{data_type}.pkl'
    version_path = os.path.join(output_dir, version_file)
    
    if os.path.exists(version_path):
        return version_path
        
    # If embeddings file doesn't exist, generate it using unpickler
    print(f"\nEmbeddings file not found for patient {patient_id_str}. Generating it now...")
    
    try:
        # Get all files for this patient
        pattern = os.path.join('source_pickles', animal,
                             f'{window_length}win{stride_length}str',
                             data_type, f'{patient_id_str}_*.pkl')
        patient_files = glob.glob(pattern)
        
        if not patient_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
            
        # Run unpickler to generate embeddings file
        version_path = unpack_single_patient(patient_files, animal, patient_id_str, 
                                           window_length, stride_length, data_type)
        print(f"Successfully generated embeddings file at {version_path}")
        return version_path
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings file: {str(e)}")

def apply_pacmap_and_clustering(embeddings, do_10d=False, 
                              mn_ratio=12.0, fp_ratio=1.0, n_neighbors=None,
                              lr=0.01):
    """Apply PaCMAP dimensionality reduction and HDBSCAN clustering.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        do_10d: whether to also compute 10D reduction (used for clustering)
        mn_ratio: PaCMAP MN_ratio parameter
        fp_ratio: PaCMAP FP_ratio parameter
        n_neighbors: number of neighbors for PaCMAP (None for auto)
        lr: learning rate for PaCMAP optimization
        
    Returns:
        tuple of (2D embeddings, 10D embeddings if do_10d=True else None, cluster labels if do_10d=True else None, pacmap_2d instance)
    """
    # Prepare PaCMAP parameters
    pacmap_params = {
        'n_components': 2,
        'MN_ratio': mn_ratio,
        'FP_ratio': fp_ratio,
        'distance': 'angular',
        'verbose': True,
        'lr': lr
    }
    
    # Only add n_neighbors if it's provided
    if n_neighbors is not None:
        pacmap_params['n_neighbors'] = n_neighbors

    # Compute 10D embeddings if requested (used for clustering)
    dim10_space = None
    cluster_labels = None
    if do_10d:
        print("\nReducing to 10 dimensions using PaCMAP...")
        pacmap_10d = PaCMAP(**{**pacmap_params, 'n_components': 10})
        dim10_space = pacmap_10d.fit_transform(embeddings)
        
        print("\nPerforming HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=100)
        cluster_labels = clusterer.fit_predict(dim10_space)
    
    # Compute 2D embeddings
    print("\nReducing to 2 dimensions using PaCMAP...")
    pacmap_2d = PaCMAP(**pacmap_params)
    dim2_space = pacmap_2d.fit_transform(embeddings)
    
    return dim2_space, dim10_space, cluster_labels, pacmap_2d

def get_param_suffix(mn_ratio, fp_ratio, lr, n_neighbors):
    """Generate filename suffix based on parameters."""
    nn_str = f"NN{n_neighbors}" if n_neighbors is not None else "NN0"
    return f"_MN{mn_ratio}_FP{fp_ratio}_LR{lr}_{nn_str}"

def process_single_patient(animal, patient_id, window_length=60, stride_length=30, 
                         data_type='train', do_10d=False, mn_ratio=12.0, 
                         fp_ratio=1.0, n_neighbors=None, lr=0.01,
                         visualize_seizures=False):
    """Process embeddings for a single patient."""
    print("\n=== Processing Brain State Embeddings ===\n")
    output_dir = setup_output_directory(animal, patient_id)
    
    # Load embeddings data
    print("\nLoading embeddings from unpickler output...")
    embeddings_path = find_embeddings_file(animal, patient_id, window_length, 
                                         stride_length, data_type)
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get the embeddings and reshape them
    embeddings_data = data['patient_embeddings']
    print(f"Original embeddings shape: {embeddings_data.shape}")  # (n_files, n_timepoints, n_features)
    
    # Reshape to (n_files*n_timepoints, n_features)
    n_files, n_timepoints, n_features = embeddings_data.shape
    embeddings_flat = embeddings_data.reshape(-1, n_features)  # Now (n_files*n_timepoints, n_features)
    print(f"Reshaped embeddings: {embeddings_flat.shape}")
    
    # Apply PaCMAP and clustering
    dim2_space, dim10_space, cluster_labels, pacmap_2d = apply_pacmap_and_clustering(
        embeddings_flat, do_10d=do_10d,
        mn_ratio=mn_ratio, fp_ratio=fp_ratio, n_neighbors=n_neighbors,
        lr=lr
    )
    
    print(f"PaCMAP output shape: {dim2_space.shape}")
    print(f"PaCMAP range - X: [{dim2_space[:, 0].min():.2f}, {dim2_space[:, 0].max():.2f}], "
          f"Y: [{dim2_space[:, 1].min():.2f}, {dim2_space[:, 1].max():.2f}]")
    
    # Save visualization
    plt.figure(figsize=(12, 10))
    
    # Get seizure labels if they exist and visualization is requested
    seizure_labels = None
    if visualize_seizures and 'seizure_labels' in data:
        seizure_labels = np.repeat(data['seizure_labels'], n_timepoints)
        
        # Plot non-seizure points first
        non_seizure_mask = seizure_labels == 0
        plt.scatter(dim2_space[non_seizure_mask, 0], 
                   dim2_space[non_seizure_mask, 1],
                   c='lightgray', s=1, alpha=0.5, label='Non-seizure')
        
        # Plot seizure points on top
        seizure_mask = seizure_labels == 1
        if np.any(seizure_mask):
            plt.scatter(dim2_space[seizure_mask, 0],
                       dim2_space[seizure_mask, 1],
                       c='red', s=2, alpha=0.8, label='Seizure')
            plt.legend()
    else:
        if do_10d:
            plt.scatter(dim2_space[:, 0], dim2_space[:, 1], c=cluster_labels, cmap='Spectral', s=1)
            plt.colorbar(label='Cluster')
        else:
            # Create a color gradient based on time points within each file
            colors = np.tile(np.arange(n_timepoints), n_files)
            plt.scatter(dim2_space[:, 0], dim2_space[:, 1], 
                       c=colors, cmap='viridis', s=1, alpha=0.5)
            plt.colorbar(label='Timepoint within window')
    
    plt.title(f'Brain State Embeddings for Patient {patient_id}\nMN={mn_ratio}, FP={fp_ratio}, n={n_neighbors}')
    plt.xlabel('PaCMAP Dimension 1')
    plt.ylabel('PaCMAP Dimension 2')
    
    # Generate parameter suffix for filenames
    param_suffix = get_param_suffix(mn_ratio, fp_ratio, lr, n_neighbors)
    
    # Save plot with parameters in filename
    plot_filename = 'tagged_pointcloud' if visualize_seizures else 'pointcloud'
    plot_path = os.path.join(output_dir, f'{plot_filename}_Epat{patient_id}{param_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save processed data
    output_data = {
        'patient_id': patient_id,
        'transformed_points_2d': dim2_space,
        'transformed_points_10d': dim10_space if do_10d else None,
        'cluster_labels': cluster_labels if do_10d else None,
        'file_indices': np.repeat(np.arange(n_files), n_timepoints),
        'window_indices': np.tile(np.arange(n_timepoints), n_files),
        'start_times': np.repeat(data['start_times'], n_timepoints),
        'stop_times': np.repeat(data['stop_times'], n_timepoints),
        'original_shape': embeddings_data.shape,
        'seizure_labels': seizure_labels,
        'pacmap_params': {
            'mn_ratio': mn_ratio,
            'fp_ratio': fp_ratio,
            'n_neighbors': n_neighbors,
            'do_10d': do_10d,
            'window_length': window_length,
            'stride_length': stride_length,
            'data_type': data_type
        },
        'pacmap_instance': pacmap_2d  # Save the PaCMAP instance
    }
    
    # Save processed data with parameters in filename
    output_path = os.path.join(output_dir, f'manifold_Epat{patient_id}{param_suffix}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nProcessing complete. Files saved to {output_dir}")
    return output_path, plot_path

def process_all_patients(animal, window_length=60, stride_length=30, data_type='train',
                        do_10d=False, mn_ratio=12.0, fp_ratio=1.0, 
                        n_neighbors=None, lr=0.01, visualize_seizures=False):
    """Process all patients that have embeddings files."""
    print("\n=== Processing All Patients ===\n")
    
    # Check output directory
    output_dir = os.path.join('output', animal)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory")
    
    # First, try to find existing embeddings files
    embeddings_files = []
    for patient_dir in glob.glob(os.path.join(output_dir, 'Epat*')):
        pattern = os.path.join(patient_dir, f'embeddings_Epat*_{window_length}win{stride_length}str_{data_type}.pkl')
        files = glob.glob(pattern)
        embeddings_files.extend(files)
    
    # If no embeddings files found, try to generate them using unpickler_split
    if not embeddings_files:
        print(f"No existing embeddings found. Attempting to generate them...")
        try:
            # Get all patient files from source directory
            source_pattern = os.path.join('source_pickles', animal,
                                        f'{window_length}win{stride_length}str',
                                        data_type, 'Epat*.pkl')
            source_files = glob.glob(source_pattern)
            
            if not source_files:
                print(f"No source files found matching pattern: {source_pattern}")
                return
            
            # Group files by patient
            patient_files = {}
            for file in source_files:
                filename = os.path.basename(file)
                match = re.match(r'(Epat\d+)_.*\.pkl', filename)
                if match:
                    patient_id = match.group(1)
                    if patient_id not in patient_files:
                        patient_files[patient_id] = []
                    patient_files[patient_id].append(file)
            
            # Process each patient's files
            for patient_id, files in patient_files.items():
                try:
                    output_path = unpack_single_patient(files, animal, patient_id,
                                                      window_length, stride_length, data_type)
                    embeddings_files.append(output_path)
                except Exception as e:
                    print(f"Error processing {patient_id}: {e}")
                    continue
            
            if not embeddings_files:
                print("Failed to generate any embeddings files")
                return
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return
    
    print(f"\nFound {len(embeddings_files)} patients to process")
    
    for embeddings_file in embeddings_files:
        try:
            # Extract patient number from filename
            match = re.search(r'Epat(\d+)', embeddings_file)
            if not match:
                continue
            patient_num = int(match.group(1))
            
            print(f"\n=== Processing Epat{patient_num} ===")
            process_single_patient(animal, patient_num, 
                                window_length=window_length,
                                stride_length=stride_length,
                                data_type=data_type,
                                do_10d=do_10d,
                                mn_ratio=mn_ratio, 
                                fp_ratio=fp_ratio,
                                n_neighbors=n_neighbors, 
                                lr=lr,
                                visualize_seizures=visualize_seizures)
        except Exception as e:
            print(f"Error processing patient {patient_num}: {e}")
            continue

def process_merged_patients(animal, patient_ids, window_length=60, stride_length=30, 
                          data_type='train', do_10d=False, mn_ratio=12.0, 
                          fp_ratio=1.0, n_neighbors=None, lr=0.01,
                          visualize_seizures=False):
    """Process and merge embeddings from multiple patients."""
    print("\n=== Merging Patient Embeddings ===\n")
    
    # Create output directory using concatenated IDs
    output_dir = os.path.join('output', animal, '_'.join([f"Epat{pid}" for pid in patient_ids]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize lists to store merged data
    merged_data = {
        'patient_embeddings': [],
        'patient_ids': [],
        'file_indices': [],
        'window_indices': [],
        'start_times': [],
        'stop_times': [],
        'seizure_labels': []
    }
    
    # Load and merge data from each patient
    total_files = 0
    for patient_id in patient_ids:
        print(f"\nLoading data for patient {patient_id}...")
        embeddings_path = find_embeddings_file(animal, patient_id, window_length, 
                                             stride_length, data_type)
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        # Get embeddings and reshape
        embeddings = data['patient_embeddings']
        print(f"Original embeddings shape for patient {patient_id}: {embeddings.shape}")
        
        # Reshape to (n_files*n_timepoints, n_features)
        n_files, n_timepoints, n_features = embeddings.shape
        embeddings_flat = embeddings.reshape(-1, n_features)
        print(f"Reshaped embeddings: {embeddings_flat.shape}")
        
        # Append data
        merged_data['patient_embeddings'].append(embeddings_flat)
        merged_data['patient_ids'].extend([f"Epat{patient_id}"] * (n_files * n_timepoints))
        merged_data['file_indices'].extend(np.repeat(np.arange(n_files) + total_files, n_timepoints))
        merged_data['window_indices'].extend(np.tile(np.arange(n_timepoints), n_files))
        merged_data['start_times'].extend(np.repeat(data['start_times'], n_timepoints))
        merged_data['stop_times'].extend(np.repeat(data['stop_times'], n_timepoints))
        
        # Append seizure labels if they exist
        if 'seizure_labels' in data and data['seizure_labels'] is not None:
            seizure_labels = np.repeat(data['seizure_labels'], n_timepoints)
        else:
            seizure_labels = np.zeros(n_files * n_timepoints, dtype=int)
        merged_data['seizure_labels'].extend(seizure_labels)
        
        total_files += n_files
        print(f"Added {n_files * n_timepoints} points")
    
    # Convert lists to arrays where appropriate
    merged_data['patient_embeddings'] = np.vstack(merged_data['patient_embeddings'])
    merged_data['seizure_labels'] = np.array(merged_data['seizure_labels'])
    print(f"\nTotal merged embeddings shape: {merged_data['patient_embeddings'].shape}")
    
    # Apply PaCMAP and clustering
    dim2_space, dim10_space, cluster_labels, pacmap_2d = apply_pacmap_and_clustering(
        merged_data['patient_embeddings'], do_10d=do_10d,
        mn_ratio=mn_ratio, fp_ratio=fp_ratio, n_neighbors=n_neighbors,
        lr=lr
    )
    
    print(f"PaCMAP output shape: {dim2_space.shape}")
    print(f"PaCMAP range - X: [{dim2_space[:, 0].min():.2f}, {dim2_space[:, 0].max():.2f}], "
          f"Y: [{dim2_space[:, 1].min():.2f}, {dim2_space[:, 1].max():.2f}]")
    
    # Save visualization
    plt.figure(figsize=(12, 10))
    
    if visualize_seizures:
        # Plot non-seizure points first
        non_seizure_mask = merged_data['seizure_labels'] == 0
        plt.scatter(dim2_space[non_seizure_mask, 0], 
                   dim2_space[non_seizure_mask, 1],
                   c='lightgray', s=1, alpha=0.5, label='Non-seizure')
        
        # Plot seizure points on top
        seizure_mask = merged_data['seizure_labels'] == 1
        if np.any(seizure_mask):
            plt.scatter(dim2_space[seizure_mask, 0],
                       dim2_space[seizure_mask, 1],
                       c='red', s=2, alpha=0.8, label='Seizure')
            plt.legend()
    elif do_10d:
        plt.scatter(dim2_space[:, 0], dim2_space[:, 1], c=cluster_labels, cmap='Spectral', s=1)
        plt.colorbar(label='Cluster')
    else:
        # Color points by patient
        unique_patients = sorted(set(merged_data['patient_ids']))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_patients)))
        
        for idx, (pat_id, color) in enumerate(zip(unique_patients, colors)):
            mask = np.array(merged_data['patient_ids']) == pat_id
            plt.scatter(dim2_space[mask, 0], 
                      dim2_space[mask, 1], 
                      color=color, 
                      label=pat_id,
                      s=1,
                      alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    merged_name = "_".join([f"Epat{pid}" for pid in patient_ids])
    plt.title(f'Brain State Embeddings for Merged Patients: {merged_name}\nMN={mn_ratio}, FP={fp_ratio}, n={n_neighbors}')
    plt.xlabel('PaCMAP Dimension 1')
    plt.ylabel('PaCMAP Dimension 2')
    
    # Generate parameter suffix for filenames
    param_suffix = get_param_suffix(mn_ratio, fp_ratio, lr, n_neighbors)
    
    # Save plot with parameters in filename
    plot_filename = 'tagged_pointcloud' if visualize_seizures else 'pointcloud'
    plot_path = os.path.join(output_dir, f'{plot_filename}_{merged_name}{param_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save processed data
    output_data = {
        'patient_ids': merged_data['patient_ids'],
        'transformed_points_2d': dim2_space,
        'transformed_points_10d': dim10_space if do_10d else None,
        'cluster_labels': cluster_labels if do_10d else None,
        'file_indices': merged_data['file_indices'],
        'window_indices': merged_data['window_indices'],
        'start_times': merged_data['start_times'],
        'stop_times': merged_data['stop_times'],
        'seizure_labels': merged_data['seizure_labels'],
        'pacmap_params': {
            'mn_ratio': mn_ratio,
            'fp_ratio': fp_ratio,
            'n_neighbors': n_neighbors,
            'do_10d': do_10d,
            'window_length': window_length,
            'stride_length': stride_length,
            'data_type': data_type
        },
        'pacmap_instance': pacmap_2d  # Save the PaCMAP instance
    }
    
    # Save processed data with parameters in filename
    output_path = os.path.join(output_dir, f'manifold_{merged_name}{param_suffix}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nProcessing complete. Files saved to {output_dir}")
    return output_path, plot_path

def main():
    parser = argparse.ArgumentParser(description='Process brain state embeddings for a specific patient.')
    parser.add_argument('--animal', type=str, required=True,
                      help='Animal name (e.g., rhesusmonkey)')
    parser.add_argument('--patient_id', type=int, help='Patient ID (e.g., 37)')
    parser.add_argument('--all', action='store_true', help='Process all patients')
    parser.add_argument('--merge', type=int, nargs='+', help='List of patient IDs as integers (e.g., 37 38)')
    parser.add_argument('--window_length', type=int, default=60,
                      help='Window length in seconds (default: 60)')
    parser.add_argument('--stride_length', type=int, default=30,
                      help='Stride length in seconds (default: 30)')
    parser.add_argument('--data_type', type=str, default='train',
                      choices=['train', 'valfinetune', 'valunseen'],
                      help='Data type to process (default: train)')
    parser.add_argument('--n_neighbors', type=int, help='Number of neighbors for PaCMAP (default: auto)')
    parser.add_argument('--do_10d', action='store_true', help='Perform 10D reduction and clustering')
    parser.add_argument('--mn_ratio', type=float, default=12.0,
                      help='PaCMAP MN_ratio parameter (default: 12.0)')
    parser.add_argument('--fp_ratio', type=float, default=1.0,
                      help='PaCMAP FP_ratio parameter (default: 1.0)')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Learning rate for PaCMAP optimization (default: 0.01)')
    parser.add_argument('--visualize_seizures', action='store_true',
                      help='Color code points based on seizure labels')
    
    args = parser.parse_args()
    
    # Common parameters for all processing functions
    common_params = {
        'window_length': args.window_length,
        'stride_length': args.stride_length,
        'data_type': args.data_type,
        'do_10d': args.do_10d,
        'mn_ratio': args.mn_ratio,
        'fp_ratio': args.fp_ratio,
        'n_neighbors': args.n_neighbors,
        'lr': args.lr,
        'visualize_seizures': args.visualize_seizures
    }
    
    try:
        if args.merge:
            process_merged_patients(args.animal, args.merge, **common_params)
        elif args.all:
            process_all_patients(args.animal, **common_params)
        elif args.patient_id:
            process_single_patient(args.animal, args.patient_id, **common_params)
        else:
            print("Error: Please specify either --patient_id, --all, or --merge")
            return
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)

if __name__ == "__main__":
    main()