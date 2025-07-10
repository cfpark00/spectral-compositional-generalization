"""
Training utilities and logic.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os
from typing import Tuple, List, Dict, Any
from tqdm import tqdm

from .models import create_model


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


def create_model_from_config(config: dict, device: torch.device) -> nn.Module:
    """Create model from configuration."""
    model = create_model(
        config["model_type"],
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_dims=config["hidden_dims"],
        dropout_rate=config["dropout_rate"],
        activation=config["activation"]
    ).to(device)
    return model


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for spectra, abundances in train_loader:
        spectra = spectra.to(device)
        abundances = abundances.to(device)
        
        optimizer.zero_grad()
        predictions = model(spectra)
        loss = criterion(predictions, abundances)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for spectra, abundances in val_loader:
            spectra = spectra.to(device)
            abundances = abundances.to(device)
            
            predictions = model(spectra)
            loss = criterion(predictions, abundances)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(config: dict, train_loader: DataLoader, val_loader: DataLoader, 
                device: torch.device, train_dir: str, test_spectra: torch.Tensor, 
                test_abundances: torch.Tensor) -> nn.Module:
    """Train the model with new checkpointing system."""
    # Create model
    model = create_model_from_config(config, device)
    
    # Create optimizer and loss
    if config.get("optimizer", "adamw").lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss(reduction='none')  # No reduction for per-sample losses
    
    # Setup directories
    logs_dir = os.path.join(train_dir, "logs")
    ckpts_dir = os.path.join(train_dir, "ckpts")
    
    # Calculate checkpoint schedule
    total_steps = len(train_loader) * config["epochs"]
    if config.get("checkpoint_by", "ratio") == "ratio":
        checkpoint_interval = int(total_steps * config.get("checkpoint_ratio", 0.05))
    else:  # step
        checkpoint_interval = config.get("checkpoint_every_n_step", 100)
    
    # Training tracking
    train_losses = []
    val_losses = []
    step = 0
    checkpoint_num = 0
    
    print("Starting training...")
    print(f"Total steps: {total_steps}, Checkpoint every: {checkpoint_interval} steps")
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(config["epochs"]), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        epoch_train_losses = []
        
        # Create batch progress bar
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", position=1, leave=False)
        
        for batch_idx, (spectra, abundances) in enumerate(batch_pbar):
            spectra = spectra.to(device)
            abundances = abundances.to(device)
            
            optimizer.zero_grad()
            predictions = model(spectra)
            losses = criterion(predictions, abundances)
            loss = losses.mean()  # Average for backprop
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
            step += 1
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Step': f'{step}/{total_steps}'})
            
            # Log every n steps (but less verbose now due to progress bars)
            if step % (config.get("log_every_n_step", 1) * 10) == 0:  # Log 10x less frequently
                tqdm.write(f"Step {step}/{total_steps}, Epoch {epoch+1}/{config['epochs']}, "
                          f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Checkpoint every n steps
            if step % checkpoint_interval == 0 or step == total_steps:
                checkpoint_num += 1
                
                # Validate
                model.eval()
                val_losses_batch = []
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc="Validating", position=2, leave=False)
                    for val_spectra, val_abundances in val_pbar:
                        val_spectra = val_spectra.to(device)
                        val_abundances = val_abundances.to(device)
                        val_predictions = model(val_spectra)
                        val_losses = criterion(val_predictions, val_abundances)
                        val_losses_batch.extend(val_losses.mean(dim=1).cpu().numpy())
                    val_pbar.close()
                
                avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
                avg_val_loss = sum(val_losses_batch) / len(val_losses_batch)
                
                # Save checkpoint
                save_checkpoint(model, optimizer, checkpoint_num, step, epoch, batch_idx,
                              avg_train_loss, avg_val_loss, ckpts_dir, config, device,
                              test_spectra, test_abundances, criterion)
                
                tqdm.write(f"Checkpoint {checkpoint_num} saved - Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}")
                
                model.train()  # Back to training mode
        
        # Store epoch losses and update epoch progress bar
        epoch_avg_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(epoch_avg_loss)
        epoch_pbar.set_postfix({'Avg Loss': f'{epoch_avg_loss:.4f}', 'Checkpoints': checkpoint_num})
        
        batch_pbar.close()
    
    epoch_pbar.close()
    
    print("Training completed!")
    
    # Save final training log
    training_log = {
        'total_steps': total_steps,
        'total_checkpoints': checkpoint_num,
        'train_losses_by_epoch': train_losses,
        'config': config
    }
    
    with open(os.path.join(logs_dir, "training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    return model


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, checkpoint_num: int,
                   step: int, epoch: int, batch_idx: int, train_loss: float, val_loss: float,
                   ckpts_dir: str, config: dict, device: torch.device,
                   test_spectra: torch.Tensor, test_abundances: torch.Tensor,
                   criterion: nn.Module):
    """Save checkpoint with model, stats, and test predictions."""
    ckpt_dir = os.path.join(ckpts_dir, f"ckpt-{checkpoint_num}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save PyTorch checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'batch_idx': batch_idx,
    }, os.path.join(ckpt_dir, "model.pt"))
    
    # Save stats
    stats = {
        'checkpoint_num': checkpoint_num,
        'step': step,
        'epoch': epoch,
        'batch_idx': batch_idx,
        'train_loss': train_loss,
        'test_loss': val_loss,
    }
    
    with open(os.path.join(ckpt_dir, "stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Get test predictions
    model.eval()
    with torch.no_grad():
        test_spectra_device = test_spectra.to(device)
        test_predictions = model(test_spectra_device).cpu()
    
    # Save test predictions as CSV
    import pandas as pd
    
    # Convert predictions to numpy and create DataFrame
    predictions_df = pd.DataFrame(
        test_predictions.numpy(),
        columns=[f'component_{i}' for i in range(test_predictions.shape[1])]
    )
    predictions_df.to_csv(os.path.join(ckpt_dir, "test_predictions.csv"), index=False)