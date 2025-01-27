import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
import torch.nn.utils as utils
from pathlib import Path
from glob import glob
import random
from Transformer import Transformer, ModelArgs
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn

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
    def __init__(self, embeddings_files, sequence_length=10, seizure_ratio=None, current_epoch=0):
        self.sequence_length = sequence_length
        self.data = []
        self.seizure_labels = []
        self.embedding_dim = None  # Will be set from the data
        
        # Load and process all embeddings
        for file_path in embeddings_files:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            embeddings = data['patient_embeddings']  # Shape: (n_files, n_timepoints, n_features)
            
            # Set embedding dimension if not set
            if self.embedding_dim is None:
                self.embedding_dim = embeddings.shape[-1]
            elif self.embedding_dim != embeddings.shape[-1]:
                raise ValueError(f"Inconsistent embedding dimensions across files: {self.embedding_dim} vs {embeddings.shape[-1]}")
                
            seizure_labels = data.get('seizure_labels', None)
            
            # Replace NaN values with zeros before normalization
            embeddings = np.nan_to_num(embeddings, nan=0.0)
            
            # Normalize embeddings with robust statistics
            mean = np.nanmean(embeddings, axis=(0, 1), keepdims=True)
            std = np.nanstd(embeddings, axis=(0, 1), keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)  # Prevent division by very small numbers
            embeddings = (embeddings - mean) / std
            
            # Flatten the first two dimensions (files and timepoints)
            n_files, n_timepoints, n_features = embeddings.shape
            embeddings_flat = embeddings.reshape(-1, n_features)
            
            # Get sequences of consecutive windows
            total_windows = len(embeddings_flat)
            for i in range(0, total_windows - sequence_length + 1):
                sequence = embeddings_flat[i:i + sequence_length]
                if not np.any(np.isnan(sequence)):  # Only add sequences without NaN
                    self.data.append(sequence)
                    
                    if seizure_labels is not None:
                        # Flatten seizure labels if they exist
                        seizure_labels_flat = np.array(seizure_labels).reshape(-1)
                        label_sequence = seizure_labels_flat[i:i + sequence_length]
                        # Check if any of the last 5 states has a seizure
                        has_seizure_in_last_5 = int(np.any(label_sequence[-5:] == 1))
                        self.seizure_labels.append(has_seizure_in_last_5)
        
        if not self.data:
            raise ValueError("No valid sequences found in the dataset")
            
        self.data = [torch.FloatTensor(seq) for seq in self.data]
        if self.seizure_labels:
            self.seizure_labels = torch.LongTensor(self.seizure_labels)
            
            if seizure_ratio is not None:
                # Early return if no seizure sequences
                seizure_indices = [i for i, label in enumerate(self.seizure_labels) if label == 1]
                if not seizure_indices:
                    logging.warning("No seizure sequences found in the dataset")
                    return
                
                # Implement smoother ratio increase over first 10 epochs
                if current_epoch < 10:
                    effective_ratio = seizure_ratio * (current_epoch + 1) / 10
                else:
                    effective_ratio = seizure_ratio
                
                # Separate sequences by seizure/non-seizure
                non_seizure_indices = [i for i, label in enumerate(self.seizure_labels) if label == 0]
                
                # Calculate target numbers
                total_sequences = len(self.data)
                target_seizure_sequences = int(total_sequences * effective_ratio)
                target_non_seizure_sequences = total_sequences - target_seizure_sequences
                
                # Sample non-seizure sequences
                sampled_non_seizure_indices = random.sample(non_seizure_indices, min(len(non_seizure_indices), target_non_seizure_sequences))
                
                # Upsample seizure sequences with conservative noise
                sampled_seizure_indices = []
                while len(sampled_seizure_indices) < target_seizure_sequences:
                    idx = seizure_indices[len(sampled_seizure_indices) % len(seizure_indices)]
                    sampled_seizure_indices.append(idx)
                
                # Combine and shuffle indices
                sampled_indices = sampled_seizure_indices + sampled_non_seizure_indices
                random.shuffle(sampled_indices)
                
                # Create new data list with conservative noise for seizure sequences
                new_data = []
                new_labels = []
                for idx in sampled_indices:
                    sequence = self.data[idx].clone()
                    
                    if idx in seizure_indices and sampled_seizure_indices.count(idx) > 1:
                        # Add very small noise (0.1% of sequence std)
                        sequence_std = torch.std(sequence)
                        noise_scale = sequence_std * 0.001  # Reduced from 5% to 0.1%
                        noise = torch.randn_like(sequence) * noise_scale
                        sequence += noise
                        
                        # Verify no NaN values were introduced
                        if torch.isnan(sequence).any():
                            sequence = self.data[idx].clone()  # Use original if noise introduced NaN
                    
                    new_data.append(sequence)
                    new_labels.append(self.seizure_labels[idx])
                
                self.data = new_data
                self.seizure_labels = torch.stack(new_labels)
                
                # Log dataset statistics
                actual_ratio = sum(l == 1 for l in self.seizure_labels) / len(self.seizure_labels)
                logging.info(f"Epoch {current_epoch}: Target ratio {effective_ratio:.3f}, Actual ratio {actual_ratio:.3f}")
                logging.info(f"Dataset size: {len(self.data)} sequences ({sum(l == 1 for l in self.seizure_labels)} seizure)")
                
                # Verify final dataset integrity
                for seq in self.data:
                    if torch.isnan(seq).any():
                        raise ValueError("NaN values detected in final dataset")
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
    
    # Remove Epat27 if present
    if 'Epat27' in patient_ids:
        patient_ids.remove('Epat27')
        logging.info("Excluded Epat27 from analysis")
    
    if not patient_ids:
        raise ValueError("No patient directories found in jackal subdirectory")
    
    # Shuffle patients
    random.shuffle(patient_ids)
    
    # Calculate split indices ensuring at least one validation patient
    n_patients = len(patient_ids)
    n_train = max(1, min(int(n_patients * train_ratio), n_patients - 2))  # Leave room for val and test
    n_val = max(1, min(int(n_patients * val_ratio), n_patients - n_train - 1))  # Ensure at least 1 val patient
    
    # Split patient IDs
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    
    logging.info(f"Total patients (excluding Epat27): {len(patient_ids)}")
    logging.info(f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    logging.info(f"Validation patients: {', '.join(val_ids)}")
    
    return train_ids, val_ids, test_ids

def get_embeddings_files(data_dir, patient_ids):
    """Get all embeddings files for given patient IDs."""
    files = []
    for pid in patient_ids:
        pattern = os.path.join(data_dir, 'jackal', pid, 'embeddings_*.pkl')
        files.extend(glob(pattern))
    return files

def check_nan_loss(loss, epoch, batch_idx):
    """Check if loss is NaN and log relevant information."""
    if torch.isnan(loss):
        logging.error(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
        return True
    return False

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, max_grad_norm=0.1, accumulation_steps=4, warmup_steps=1000, total_steps=0):
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    for batch_idx, batch in enumerate(train_loader):
        if isinstance(batch, tuple):
            sequences, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
        else:
            sequences = batch.to(device)
            labels = None
            
        # Log input statistics periodically
        if batch_idx % 100 == 0:
            with torch.no_grad():
                seq_mean = sequences.mean().item()
                seq_std = sequences.std().item()
                seq_min = sequences.min().item()
                seq_max = sequences.max().item()
                logging.info(f"Input stats - Mean: {seq_mean:.3f}, Std: {seq_std:.3f}, Range: [{seq_min:.3f}, {seq_max:.3f}]")
        
        batch_size = sequences.size(0)
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda', enabled=True):
            # Pass seizure labels to model for attention bias
            output = model(sequences, start_pos=0, seizure_labels=labels)
            loss = criterion(output[:, :-1, :], sequences[:, 1:, :], labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation
        
        # Check for NaN loss
        if check_nan_loss(loss, epoch, batch_idx):
            return float('nan')
        
        # Backward pass
        loss.backward()
        
        # Only step optimizer and zero gradients after accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Clip gradients
            utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Apply learning rate warmup
            if total_steps < warmup_steps:
                lr_scale = min(1., float(total_steps + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * 1e-5  # Scale up to 1e-5
            
            # Step optimizer
            optimizer.step()
            optimizer.zero_grad()
            
            # Log gradient norms periodically
            if batch_idx % 100 == 0:
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                logging.info(f"Gradient norm after clipping: {grad_norm:.6f}")
        
        total_loss += (loss.item() * accumulation_steps) * batch_size
        total_steps += 1
        
        if batch_idx % 100 == 0:
            logging.info(f'Batch {batch_idx}/{len(train_loader)}, Cosine Loss: {loss.item() * accumulation_steps:.6f}')
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch, tuple):
                sequences, labels = batch
                sequences = sequences.to(device)
                labels = labels.to(device)
            else:
                sequences = batch.to(device)
                labels = None
            
            batch_size = sequences.size(0)
            
            # Debug logging
            if batch_idx == 0:  # Log first batch stats
                logging.info(f"Validation batch stats - Shape: {sequences.shape}, Mean: {sequences.mean():.6f}, Std: {sequences.std():.6f}")
                logging.info(f"Range - Min: {sequences.min():.6f}, Max: {sequences.max():.6f}")
            
            # Forward pass
            output = model(sequences, start_pos=0, seizure_labels=None)  # Explicitly pass None for seizure_labels
            
            # Debug output stats
            if batch_idx == 0:  # Log first batch output stats
                logging.info(f"Model output stats - Shape: {output.shape}, Mean: {output.mean():.6f}, Std: {output.std():.6f}")
                logging.info(f"Output range - Min: {output.min():.6f}, Max: {output.max():.6f}")
            
            # Calculate loss (predicting next timestep) - Note: no seizure labels needed for standard MSE
            loss = criterion(output[:, :-1, :], sequences[:, 1:, :])
            
            if torch.isnan(loss):
                logging.error(f"NaN loss in validation batch {batch_idx}")
                logging.error(f"Loss components - MSE inputs:")
                logging.error(f"Pred shape: {output[:, :-1, :].shape}, Target shape: {sequences[:, 1:, :].shape}")
                logging.error(f"Pred stats - Mean: {output[:, :-1, :].mean():.6f}, Std: {output[:, :-1, :].std():.6f}")
                logging.error(f"Target stats - Mean: {sequences[:, 1:, :].mean():.6f}, Std: {sequences[:, 1:, :].std():.6f}")
                continue
            
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

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_losses(train_losses, val_losses, save_dir):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Cosine Loss')
    plt.plot(val_losses, label='Validation Cosine Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Loss (1 - cos_sim)')
    plt.title('Training and Validation Cosine Losses')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_dir, f'cosine_loss_plot_{timestamp}.png'))
    plt.close()

class WeightedCosineLoss(nn.Module):
    def __init__(self, pre_ictal_weight=5.0, pre_ictal_window=10):
        super().__init__()
        self.pre_ictal_weight = pre_ictal_weight
        self.pre_ictal_window = pre_ictal_window
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(self, output, target, seizure_labels=None):
        # Calculate cosine similarity for each timestep
        # Shape: [batch_size, seq_len]
        cos_sim = self.cosine_sim(output, target)
        
        # Convert to loss (1 - cos_sim to minimize)
        base_loss = 1 - cos_sim
        
        if seizure_labels is not None:
            # Create weights tensor initialized to ones
            weights = torch.ones_like(base_loss)  # Shape: [batch_size, seq_len]
            
            # For each sequence in the batch
            for i in range(len(seizure_labels)):
                # Find indices where seizures occur
                seizure_indices = torch.where(seizure_labels[i] == 1)[0]
                
                if len(seizure_indices) > 0:
                    # For each seizure, mark the preceding window with higher weight
                    for seizure_idx in seizure_indices:
                        start_idx = max(0, seizure_idx - self.pre_ictal_window)
                        weights[i, start_idx:seizure_idx + 1] = self.pre_ictal_weight
            
            # Apply weights
            weighted_loss = base_loss * weights
            
            return weighted_loss.mean()
        
        return base_loss.mean()

def main():
    # Training settings
    config = {
        'data_dir': 'output',
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'plot_dir': 'training_plots',
        'sequence_length': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 30,
        'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        'n_layers': 8,
        'n_heads': 8,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'max_grad_norm': 0.5,
        'pre_ictal_weight': 5.0,  # Weight for pre-ictal sequences in loss
        'pre_ictal_window': 10,   # Number of states before seizure to weight
        'pre_ictal_bias': 2.0     # Attention bias for pre-ictal sequences
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train transformer model on brain state embeddings.')
    parser.add_argument('--seizure_ratio', type=float,
                      help='Ratio of training sequences that should end with seizure states (e.g., 0.2 for 20%%)')
    parser.add_argument('--pre_ictal_weight', type=float, default=5.0,
                      help='Weight to apply to pre-ictal sequence prediction errors (default: 5.0)')
    parser.add_argument('--pre_ictal_window', type=int, default=10,
                      help='Number of brain states before seizure to apply higher weight to (default: 10)')
    parser.add_argument('--pre_ictal_bias', type=float, default=2.0,
                      help='Attention bias to apply to pre-ictal sequences (default: 2.0)')
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(config['log_dir'])
    logging.info(f"Starting training with config: {json.dumps(config, indent=2)}")
    if args.seizure_ratio is not None:
        logging.info(f"Using seizure ratio: {args.seizure_ratio}")
    
    # Log device info
    logging.info(f"Using device: {config['device']}")
    if config['device'] == 'cuda':
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    elif config['device'] == 'mps':
        logging.info("Metal Performance Shaders (MPS) backend is being used for GPU acceleration")
    else:
        logging.info("Running on CPU")
    
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
    train_dataset = BrainStateDataset(train_files, config['sequence_length'], seizure_ratio=args.seizure_ratio)
    val_dataset = BrainStateDataset(val_files, config['sequence_length'])  # No seizure ratio for validation
    test_dataset = BrainStateDataset(test_files, config['sequence_length'])  # No seizure ratio for test
    
    # Get embedding dimension from dataset
    config['model_dim'] = train_dataset.embedding_dim
    logging.info(f"Using embedding dimension from data: {config['model_dim']}")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           collate_fn=collate_fn)
    
    # Initialize model with pre-ictal attention parameters
    model_args = ModelArgs(
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_batch_size=config['batch_size'],
        max_seq_len=config['sequence_length'],
        device=config['device'],
        pre_ictal_window=args.pre_ictal_window,
        pre_ictal_bias=args.pre_ictal_bias
    )
    
    model = Transformer(model_args).to(config['device'])
    
    # Log model parameters
    n_params = count_parameters(model)
    logging.info(f"Number of trainable parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    train_criterion = WeightedCosineLoss(
        pre_ictal_weight=args.pre_ictal_weight,
        pre_ictal_window=args.pre_ictal_window
    )
    val_criterion = WeightedCosineLoss(pre_ictal_weight=1.0, pre_ictal_window=10)  # No weighting for validation
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        logging.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Recreate training dataset each epoch to update seizure ratio
        if args.seizure_ratio is not None:
            train_dataset = BrainStateDataset(train_files, config['sequence_length'], 
                                            seizure_ratio=args.seizure_ratio, 
                                            current_epoch=epoch)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                    shuffle=True, collate_fn=collate_fn)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, train_criterion, 
                               config['device'], epoch, config['max_grad_norm'])
        
        # Check for NaN loss
        if np.isnan(train_loss):
            logging.error("Training stopped due to NaN loss")
            break
        
        logging.info(f"Training Loss: {train_loss:.6f}")
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, val_criterion, config['device'])
        logging.info(f"Validation Loss: {val_loss:.6f}")
        val_losses.append(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logging.info("New best model!")
        
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            config['checkpoint_dir'], is_best
        )
        
        # Plot losses every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_losses(train_losses, val_losses, config['plot_dir'])
    
    # Final loss plot
    plot_losses(train_losses, val_losses, config['plot_dir'])
    logging.info("Training completed!")

if __name__ == "__main__":
    main() 