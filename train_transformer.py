import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
from pathlib import Path
from glob import glob
import random
from Transformer import Transformer, ModelArgs
import logging
import json
from datetime import datetime

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
    def __init__(self, embeddings_files, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = []
        self.seizure_labels = []
        
        for file_path in embeddings_files:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            embeddings = data['patient_embeddings']  # Shape: (n_files, n_timepoints, n_features)
            seizure_labels = data.get('seizure_labels', None)
            
            # Flatten the first two dimensions (files and timepoints)
            n_files, n_timepoints, n_features = embeddings.shape
            embeddings_flat = embeddings.reshape(-1, n_features)
            
            # Get sequences of consecutive windows
            total_windows = len(embeddings_flat)
            for i in range(0, total_windows - sequence_length + 1):
                sequence = embeddings_flat[i:i + sequence_length]
                self.data.append(sequence)
                
                if seizure_labels is not None:
                    # Flatten seizure labels if they exist
                    seizure_labels_flat = np.array(seizure_labels).reshape(-1)
                    label_sequence = seizure_labels_flat[i:i + sequence_length]
                    # Convert to binary: 1 if any window in sequence has seizure
                    has_seizure = int(np.any(label_sequence == 1))
                    self.seizure_labels.append(has_seizure)
        
        self.data = [torch.FloatTensor(seq) for seq in self.data]
        if self.seizure_labels:
            self.seizure_labels = torch.LongTensor(self.seizure_labels)
        else:
            self.seizure_labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        if self.seizure_labels is not None:
            label = self.seizure_labels[idx]
            return sequence, label
        return sequence

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def split_patients(data_dir, train_ratio=0.7, val_ratio=0.15):
    """Split patients into train/val/test sets."""
    # Get all unique patient directories under jackal
    patient_dirs = glob(os.path.join(data_dir, 'jackal', 'Epat*'))
    patient_ids = [os.path.basename(d) for d in patient_dirs]
    
    if not patient_ids:
        raise ValueError("No patient directories found in jackal subdirectory")
    
    # Shuffle patients
    random.shuffle(patient_ids)
    
    # Calculate split indices
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split patient IDs
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    
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
    
    for batch_idx, batch in enumerate(train_loader):
        if isinstance(batch, tuple):
            sequences, labels = batch
        else:
            sequences = batch
            
        # Move data to device
        sequences = sequences.to(device)
        batch_size = sequences.size(0)
        
        # Forward pass
        output = model(sequences)
        
        # Calculate loss (predicting next timestep)
        loss = criterion(output[:, :-1, :], sequences[:, 1:, :])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        
        if batch_idx % 100 == 0:
            logging.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, tuple):
                sequences, labels = batch
            else:
                sequences = batch
            
            # Move data to device
            sequences = sequences.to(device)
            batch_size = sequences.size(0)
            
            output = model(sequences)
            loss = criterion(output[:, :-1, :], sequences[:, 1:, :])
            
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

def main():
    # Training settings
    config = {
        'data_dir': 'output',
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'sequence_length': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_dim': 512,  # Match the embedding dimension
        'n_layers': 8,
        'n_heads': 8,
        'train_ratio': 0.7,
        'val_ratio': 0.15
    }
    
    # Setup logging
    log_file = setup_logging(config['log_dir'])
    logging.info(f"Starting training with config: {json.dumps(config, indent=2)}")
    
    # Split patients
    train_ids, val_ids, test_ids = split_patients(
        config['data_dir'], 
        config['train_ratio'], 
        config['val_ratio']
    )
    logging.info(f"Train patients: {len(train_ids)}, Val patients: {len(val_ids)}, Test patients: {len(test_ids)}")
    
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
    criterion = MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        logging.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        logging.info(f"Training Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config['device'])
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
    
    logging.info("Training completed!")

if __name__ == "__main__":
    main() 