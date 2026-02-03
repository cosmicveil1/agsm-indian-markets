import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_lob import FI2010Dataset
from models.agsm_lob import AGSMNetLOB

def train_one_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = criterion(logits, y.view(-1))
        
        # Scale loss
        loss = loss / accumulation_steps
        loss.backward()
        
        # Optimize every N steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps # Unscale for logging
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y.view(-1))
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return avg_loss, acc, f1, all_targets, all_preds

def main():
    parser = argparse.ArgumentParser(description="Train AGSMNet on FI-2010 LOB Data")
    parser.add_argument('--data', type=str, required=True, help='Path to FI-2010 text file')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8, help='Reduced batch size for LOB (memory intensive)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--horizon', type=int, default=10, help='Prediction horizon (k=10, 20, 50, 100)')
    parser.add_argument('--subset', type=float, default=1.0, help='Fraction of data to use (e.g. 0.1 for 10%)')
    parser.add_argument('--dim', type=int, default=32, help='Model dimension (d_model)')
    parser.add_argument('--state_dim', type=int, default=32, help='SSM State dimension')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile optimization')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load Dataset
    # FI-2010 usually comes as one big file for the 7-day train/test standardized set
    # We will simulate a split here since we load one file
    dataset = FI2010Dataset(args.data, k=args.horizon)
    
    # Subset for faster training if requested
    if args.subset < 1.0:
        subset_size = int(len(dataset) * args.subset)
        print(f"Subsetting dataset to {args.subset*100}% ({subset_size} samples)")
        # Use random split to pick a subset, discard the rest
        dataset, _ = torch.utils.data.random_split(dataset, [subset_size, len(dataset) - subset_size])
    
    # Simple split (80/20) for demonstration if only one file provided
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = AGSMNetLOB(
        in_channels=1, 
        dim=args.dim, 
        state_dim=args.state_dim,
        num_classes=3
    ).to(args.device)
    

    
    # Optimize with torch.compile (Optional)
    if hasattr(args, 'compile') and args.compile:
        try:
            print("Compiling model with torch.compile...")
            torch.set_float32_matmul_precision('high')
            model = torch.compile(model)
            print("Model compiled.")
        except Exception as e:
            print(f"Compilation failed: {e}")
    else:
        print("Skipping torch.compile (use --compile to enable). Running in eager mode.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # torch.cuda.empty_cache() # Removed to prevent sync overhead
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, 
            accumulation_steps=args.accumulation_steps
        )
        test_loss, test_acc, test_f1, _, _ = evaluate(model, test_loader, criterion, args.device)
        
        scheduler.step(test_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}  | Acc: {test_acc*100:.2f}% | F1: {test_f1:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_lob_model.pth')
            print("  ✅ New best model saved!")
            
    print(f"Training Complete. Best Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
