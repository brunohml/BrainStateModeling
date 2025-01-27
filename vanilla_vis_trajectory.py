import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pacmap import PaCMAP
import argparse
import random
from pathlib import Path
from glob import glob

def load_embeddings(patient_num):
    """Load embeddings from source pickle files for a specific patient."""
    patient_id = f"Epat{patient_num}"
    base_path = os.path.join('source_pickles', 'jackal', 'Epoch39', 
                            '60SecondWindow_30SecondStride', 'train')
    
    # Find all pickle files for this patient
    pattern = os.path.join(base_path, f"{patient_id}_*.pkl")
    files = glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No pickle files found for {patient_id}")
    
    print(f"Found {len(files)} files for {patient_id}")
    
    # Load all embeddings
    all_embeddings = []
    for file_path in files:
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
            if embeddings.shape != (32, 512):
                print(f"Warning: Unexpected shape {embeddings.shape} in {file_path}")
                continue
            all_embeddings.append(embeddings)
    
    # Stack all embeddings
    embeddings_array = np.stack(all_embeddings)
    print(f"Total shape: {embeddings_array.shape} (n_files, timesteps, features)")
    
    return embeddings_array, files

def get_random_sequence(embeddings, files):
    """Get a random sequence of 32 consecutive embeddings from a single file."""
    n_files = len(embeddings)
    
    # Pick a random file
    file_idx = random.randint(0, n_files - 1)
    sequence = embeddings[file_idx]
    
    # Calculate the start index in the flattened array
    start_idx = file_idx * 32
    
    print(f"Selected file: {os.path.basename(files[file_idx])}")
    return sequence, start_idx

def visualize_trajectory(embeddings, sequence, start_idx, output_dir):
    """Visualize the trajectory using PaCMAP."""
    N, T, F = embeddings.shape
    print(f"Original embeddings shape: (n_files={N}, timesteps={T}, features={F})")
    print(f"Visualizing file {start_idx // T} (timesteps {start_idx}-{start_idx + T - 1})")
    
    # Reshape maintaining temporal order within each file
    embeddings = embeddings.transpose(0, 2, 1).reshape(-1, F)
    print(f"Reshaped embeddings: (n_files*timesteps={embeddings.shape[0]}, features={embeddings.shape[1]})")
    
    # Initialize PaCMAP
    pacmap = PaCMAP(
        n_components=2,
        MN_ratio=2.0,
        FP_ratio=0.1,
        distance='angular',
        lr=0.01
    )
    
    # Fit PaCMAP on all embeddings
    embeddings_2d = pacmap.fit_transform(embeddings)
    
    # Get the coordinates for our sequence
    sequence_2d = embeddings_2d[start_idx:start_idx + len(sequence)]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot all points in grey
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgrey', s=1, alpha=0.1)
    
    # Create color gradient for trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, len(sequence)))
    
    # Plot trajectory points with color gradient
    for i in range(len(sequence_2d) - 1):
        # Plot line connecting consecutive points
        plt.plot(sequence_2d[i:i+2, 0], sequence_2d[i:i+2, 1], 
                c=colors[i], linewidth=2, alpha=0.7)
        # Plot point
        plt.scatter(sequence_2d[i, 0], sequence_2d[i, 1], 
                   c=[colors[i]], s=50)
    
    # Plot the last point with a star marker
    plt.scatter(sequence_2d[-1, 0], sequence_2d[-1, 1], 
               c=[colors[-1]], s=200, marker='*', label='End')
    
    plt.title('Brain State Trajectory')
    plt.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'trajectory_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize brain state trajectory using PaCMAP')
    parser.add_argument('--patient', type=int, required=True,
                      help='Patient number (without Epat prefix)')
    
    args = parser.parse_args()
    
    # Load data
    embeddings, files = load_embeddings(args.patient)
    
    # Get random sequence
    sequence, start_idx = get_random_sequence(embeddings, files)
    
    # Setup output directory
    output_dir = os.path.join('output', 'jackal', f'Epat{args.patient}')
    
    # Visualize
    visualize_trajectory(embeddings, sequence, start_idx, output_dir)
    print(f"Visualization saved to {output_dir}/trajectory_visualization.png")

if __name__ == "__main__":
    main()
