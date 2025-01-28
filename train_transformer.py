import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss, CosineSimilarity
from pathlib import Path
from glob import glob
import random
from Transformer import Transformer, ModelArgs
import logging
import json
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

def collate_fn(batch):
    """Custom collate function to handle sequences and labels."""
    sequences = []
    labels = []
    
    for item in batch:
        if isinstance(item, tuple):
            seq, label = item
            sequences.append(seq)
            labels.append(label)
        else:
            sequences.append(item)
    
    sequences = torch.stack(sequences)
    if labels:
        labels = torch.stack(labels)
        return sequences, labels
    return sequences

class BrainStateDataset(Dataset):
    def __init__(self, embeddings_files, sequence_length=16):
        self.sequence_length = sequence_length
        self.data = []
        self.seizure_labels = []
        
        total_sequences = 0
        total_files = 0
        
        for file_path in embeddings_files:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            embeddings = data['patient_embeddings']  # Shape: (n_files, n_timepoints, n_features)
            file_indices = data['file_indices']
            seizure_labels = data.get('seizure_labels', None)
            
            n_files, n_timepoints, n_features = embeddings.shape
            total_files += n_files
            
            logging.info(f"Processing file {file_path}")
            logging.info(f"Contains {n_files} files with {n_timepoints} timepoints each")
            
            # Process each file separately
            for file_idx in range(n_files):
                file_embeddings = embeddings[file_idx]  # Shape: (n_timepoints, n_features)
                
                # Create sliding windows with stride 1
                n_windows = n_timepoints - sequence_length + 1
                for window_start in range(n_windows):
                    window = file_embeddings[window_start:window_start + sequence_length]
                    self.data.append(window)
                    total_sequences += 1
                    
                    if seizure_labels is not None:
                        # Get labels for this window
                        file_labels = seizure_labels[file_idx]
                        if np.isscalar(file_labels):
                            has_seizure = int(file_labels == 1)
                            self.seizure_labels.append(has_seizure)
                        else:
                            window_labels = file_labels[window_start:window_start + sequence_length]
                            has_seizure = int(np.any(window_labels == 1))
                            self.seizure_labels.append(has_seizure)
        
        self.data = [torch.FloatTensor(seq) for seq in self.data]
        if self.seizure_labels:
            self.seizure_labels = torch.LongTensor(self.seizure_labels)
        else:
            self.seizure_labels = None
            
        logging.info(f"Created dataset with {total_sequences} sequences from {total_files} files")
        logging.info(f"Average sequences per file: {total_sequences/total_files:.1f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        if self.seizure_labels is not None:
            label = self.seizure_labels[idx]
            return sequence, label
        return sequence

def setup_logging():
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', timestamp)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_dir

def split_patients(data_dir, test_patient):
    """Split patients into train/val/test sets, using all available patients."""
    # Get all unique patient directories under jackal
    patient_dirs = glob(os.path.join(data_dir, 'jackal', 'Epat*'))
    patient_ids = [os.path.basename(d) for d in patient_dirs]
    
    if not patient_ids:
        raise ValueError("No patient directories found in jackal subdirectory")
    
    if test_patient not in patient_ids:
        raise ValueError(f"Test patient {test_patient} not found in data directory")
    
    # Remove test patient from available patients
    patient_ids.remove(test_patient)
    
    # Need at least 3 patients total (1 train, 1 val, 1 test)
    if len(patient_ids) < 2:
        raise ValueError(f"Not enough patients. Need at least 3 patients, found {len(patient_ids) + 1}")
    
    # Randomly select 1 validation patient
    val_ids = random.sample(patient_ids, 1)
    
    # Remove validation patient from remaining pool
    patient_ids.remove(val_ids[0])
    
    # All remaining patients go to training
    train_ids = patient_ids
    test_ids = [test_patient]
    
    return train_ids, val_ids, test_ids

def get_embeddings_files(data_dir, patient_ids):
    """Get all embeddings files for given patient IDs."""
    files = []
    for pid in patient_ids:
        pattern = os.path.join(data_dir, 'jackal', pid, 'embeddings_*.pkl')
        files.extend(glob(pattern))
    return files

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    
    logging.info(f"Training on {len(train_loader.dataset)} examples in {total_batches} batches")
    
    for batch_idx, batch in enumerate(train_loader):
        if isinstance(batch, tuple):
            sequences, labels = batch
        else:
            sequences = batch
            
        # Move data to device
        sequences = sequences.to(device)
        batch_size = sequences.size(0)
        
        # Forward pass
        output = model(sequences)  # Shape: [batch_size, seq_len, dim]
        
        # Calculate cosine loss between predicted and actual states
        output_flat = output.reshape(-1, output.size(-1))
        target_flat = sequences.reshape(-1, sequences.size(-1))
        
        # Calculate cosine similarity (returns values between -1 and 1)
        cos_sim = criterion(output_flat, target_flat)
        
        # Convert to loss (1 - cos_sim to minimize)
        base_loss = 1 - cos_sim
        loss = base_loss.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        
        if batch_idx % 100 == 0:
            logging.info(f'Batch {batch_idx}/{total_batches}, Loss: {loss.item():.6f}')
    
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_batches = len(val_loader)
    
    logging.info(f"Validating on {len(val_loader.dataset)} examples in {total_batches} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch, tuple):
                sequences, labels = batch
            else:
                sequences = batch
            
            # Move data to device
            sequences = sequences.to(device)
            batch_size = sequences.size(0)
            
            # Forward pass
            output = model(sequences)
            
            # Calculate cosine loss between predicted and actual states
            output_flat = output.reshape(-1, output.size(-1))
            target_flat = sequences.reshape(-1, sequences.size(-1))
            
            cos_sim = criterion(output_flat, target_flat)
            base_loss = 1 - cos_sim
            loss = base_loss.mean()
            
            total_loss += loss.item() * batch_size
    
    return total_loss / len(val_loader.dataset)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best so far
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pat', type=int, required=True, 
                       help='Patient number to use for testing (e.g., 1 for Epat1)')
    args = parser.parse_args()
    
    # Construct full patient ID
    test_patient = f"Epat{args.test_pat}"
    
    # Setup logging first
    log_dir = setup_logging()
    
    # Training settings
    config = {
        'data_dir': 'output',
        'log_dir': log_dir,  # Use the timestamped log directory
        'checkpoint_dir': os.path.join(log_dir, 'checkpoints'),  # Save checkpoints in the same directory
        'sequence_length': 16,
        'batch_size': 32,
        'learning_rate': 1e-6,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_dim': 512,  # Match the embedding dimension
        'n_layers': 8,
        'n_heads': 8
    }
    
    logging.info(f"Starting training with config: {json.dumps(config, indent=2)}")
    logging.info(f"Test patient: {test_patient}")
    
    # Split patients with fixed sizes
    train_ids, val_ids, test_ids = split_patients(config['data_dir'], test_patient)
    logging.info(f"Train patients: {train_ids}")
    logging.info(f"Val patients: {val_ids}")
    logging.info(f"Test patients: {test_ids}")
    
    
    # Get data files
    train_files = get_embeddings_files(config['data_dir'], train_ids)
    val_files = get_embeddings_files(config['data_dir'], val_ids)
    test_files = get_embeddings_files(config['data_dir'], test_ids)
    
    # Create datasets
    train_dataset = BrainStateDataset(train_files, config['sequence_length'])
    val_dataset = BrainStateDataset(val_files, config['sequence_length'])
    test_dataset = BrainStateDataset(test_files, config['sequence_length'])
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           collate_fn=collate_fn)
    
    # Initialize model
    model_args = ModelArgs(
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_batch_size=config['batch_size'],
        max_seq_len=config['sequence_length'],
        device=config['device']
    )
    
    model = Transformer(model_args).to(config['device'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = CosineSimilarity(dim=1)
    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config['num_epochs']):
        logging.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        train_losses.append(train_loss)
        logging.info(f"Training Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config['device'])
        val_losses.append(val_loss)
        logging.info(f"Validation Loss: {val_loss:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logging.info("New best model!")
        
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            config['checkpoint_dir'], is_best
        )
        
        # Update and save the loss plot after each epoch
        plot_losses(train_losses, val_losses, os.path.join(config['log_dir'], 'loss_plot.png'))
    
    logging.info("Training completed!")

if __name__ == "__main__":
    main() 