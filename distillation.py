import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import time
import logging
import matplotlib.pyplot as plt
import json

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DistillationDataset(torch.utils.data.Dataset):
    """Custom dataset for knowledge distillation with soft labels"""
    
    def __init__(self, dataset, processor, teacher_logits=None):
        self.dataset = dataset
        self.processor = processor
        self.teacher_logits = teacher_logits
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        # Ensure image is RGB (3 channels) to avoid dimension errors
        image = image.convert('RGB')
            
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        result = {
            'pixel_values': pixel_values,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }
        
        # Add teacher soft labels if available
        if self.teacher_logits is not None:
            result['teacher_logits'] = torch.tensor(self.teacher_logits[idx], dtype=torch.float32)
            
        return result

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation"""
    
    def __init__(self, alpha=0.5, temperature=3.0, use_cls_mse=False):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.use_cls_mse = use_cls_mse
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_logits, teacher_logits, labels, 
                student_cls=None, teacher_cls=None, epoch=0):
        
        # More aggressive dynamic alpha: start with strong emphasis on hard targets
        if epoch < 5:
            dynamic_alpha = 0.1  # Heavy emphasis on hard targets early
        elif epoch < 10:
            dynamic_alpha = 0.3  # Gradually introduce soft targets
        else:
            dynamic_alpha = self.alpha  # Use full alpha after epoch 10
        
        # Hard target loss (student vs ground truth)
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Soft target loss (student vs teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = self.kl_loss(student_soft, teacher_soft)
        
        # Combined loss with dynamic weighting
        total_loss = (1 - dynamic_alpha) * ce_loss + dynamic_alpha * (self.temperature ** 2) * kl_loss
        
        # Optional CLS token MSE loss
        if self.use_cls_mse and student_cls is not None and teacher_cls is not None:
            cls_loss = self.mse_loss(student_cls, teacher_cls)
            total_loss += 0.1 * cls_loss  # Small weight for CLS loss
            
        return total_loss, ce_loss, kl_loss, dynamic_alpha

def generate_teacher_predictions(teacher_model, dataloader, temperature=4.0):
    """Generate soft predictions from teacher model"""
    teacher_model.eval()
    all_logits = []
    all_cls_tokens = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating teacher predictions"):
            pixel_values = batch['pixel_values'].to(device)
            
            outputs = teacher_model(pixel_values, output_hidden_states=True)
            logits = outputs.logits
            
            # Apply temperature scaling
            soft_logits = logits / temperature
            all_logits.append(soft_logits.cpu())
            
            # Extract CLS token (first token of last hidden state)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                cls_token = outputs.hidden_states[-1][:, 0, :]  # [batch_size, hidden_dim]
                all_cls_tokens.append(cls_token.cpu())
    
    return torch.cat(all_logits, dim=0), torch.cat(all_cls_tokens, dim=0) if all_cls_tokens else None

class DistillationTrainer:
    """Custom trainer for knowledge distillation"""
    
    def __init__(self, student_model, teacher_model, train_dataloader, val_dataloader,
                 optimizer, scheduler, loss_fn, epochs=60):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        
        # Learning curve tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.start_epoch = 0
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training state"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load learning curves
        if 'learning_curves' in checkpoint:
            self.train_losses = checkpoint['learning_curves']['train_losses']
            self.train_accuracies = checkpoint['learning_curves']['train_accuracies']
            self.val_losses = checkpoint['learning_curves']['val_losses']
            self.val_accuracies = checkpoint['learning_curves']['val_accuracies']
        
        # Set starting epoch
        self.start_epoch = checkpoint['epoch']
        
        print(f"Resumed from epoch {self.start_epoch}")
        print(f"Last train accuracy: {checkpoint.get('train_acc', 'N/A'):.2f}%")
        print(f"Last validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return checkpoint
        
    def train(self):
        best_acc = 0.0
        
        # Find current best accuracy from loaded curves if resuming
        if self.val_accuracies:
            best_acc = max(self.val_accuracies)
            print(f"Current best validation accuracy: {best_acc:.2f}%")
        
        for epoch in range(self.start_epoch, self.epochs):
            # Training phase
            self.student_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                teacher_logits = batch['teacher_logits'].to(device)
                
                # Student forward pass
                student_outputs = self.student_model(pixel_values, output_hidden_states=True)
                student_logits = student_outputs.logits
                
                # Early debugging for first epoch only if starting from scratch
                if epoch == 0 and batch_idx < 3 and self.start_epoch == 0:
                    print(f"\nBatch {batch_idx}: Student logits range: [{student_logits.min():.3f}, {student_logits.max():.3f}]")
                    print(f"Teacher logits range: [{teacher_logits.min():.3f}, {teacher_logits.max():.3f}]")
                    print(f"Labels: {labels[:8].tolist()}")
                    print(f"Student predictions: {student_logits.argmax(dim=1)[:8].tolist()}")
                
                # Extract CLS tokens if using MSE loss
                student_cls = None
                if self.loss_fn.use_cls_mse and hasattr(student_outputs, 'hidden_states'):
                    student_cls = student_outputs.hidden_states[-1][:, 0, :]
                
                # Compute loss
                loss, ce_loss, kl_loss, dynamic_alpha = self.loss_fn(
                    student_logits, teacher_logits, labels, student_cls, None, epoch
                )
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(student_logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'CE': f"{ce_loss.item():.4f}",
                    'KL': f"{kl_loss.item():.4f}",
                    'Alpha': f"{dynamic_alpha:.3f}",
                    'Acc': f"{100*train_correct/train_total:.2f}%"
                })
            
            if self.scheduler:
                self.scheduler.step()
            
            # Validation phase
            val_acc, val_loss = self.evaluate()
            
            # Record learning curves
            epoch_train_loss = train_loss / len(self.train_dataloader)
            epoch_train_acc = 100 * train_correct / train_total
            
            self.train_losses.append(epoch_train_loss)
            self.train_accuracies.append(epoch_train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint every epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.student_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'train_loss': epoch_train_loss,
                'train_acc': epoch_train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_curves': {
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies
                }
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.student_model.state_dict(), 'best_distilled_model.pth')
            
            # Save learning curves plot every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_learning_curves()
                
        # Final learning curves plot and save
        self.plot_learning_curves()
        self.save_learning_curves()
                
        return best_acc
    
    def evaluate(self):
        self.student_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.student_model(pixel_values)
                logits = outputs.logits
                
                # Calculate loss
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = 100 * correct / total

        return accuracy, avg_loss
    
    def plot_learning_curves(self):
        """Plot and save learning curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_learning_curves(self):
        """Save learning curves data to JSON"""
        curves_data = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        with open('learning_curves.json', 'w') as f:
            json.dump(curves_data, f, indent=2)

def main(resume_from_checkpoint=None):
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("sitloboi2012/rvl_cdip_large_dataset")
    train_dataset = ds['train']
    val_dataset = ds['validate']
    
    # Initialize models and processor
    TEACHER_MODEL = "microsoft/dit-large-finetuned-rvlcdip"
    STUDENT_BASE = "WinKawaks/vit-tiny-patch16-224"  # Untrained ViT-Tiny base model
    
    processor = AutoImageProcessor.from_pretrained(TEACHER_MODEL)
    teacher_model = AutoModelForImageClassification.from_pretrained(TEACHER_MODEL).to(device)
    
    # Initialize ViT-Tiny student from scratch with correct number of classes
    student_model = AutoModelForImageClassification.from_pretrained(
        STUDENT_BASE, 
        num_labels=16,  # RVL-CDIP has 16 classes
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Better weight initialization for faster convergence (only if not resuming)
    if not resume_from_checkpoint:
        for name, param in student_model.named_parameters():
            if 'classifier' in name or 'head' in name:
                if param.dim() > 1:
                    torch.nn.init.xavier_normal_(param, gain=0.02)
                else:
                    torch.nn.init.zeros_(param)
    
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    print(f"Compression ratio: {sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()):.1f}x")
    
    # Create initial datasets for teacher prediction generation (only if not resuming or teacher logits don't exist)
    teacher_logits_file = 'teacher_logits.pt'
    if resume_from_checkpoint or not hasattr(main, '_teacher_logits_generated'):
        # Try to load existing teacher logits to save time
        try:
            print("Loading existing teacher logits...")
            teacher_logits = torch.load(teacher_logits_file)
            teacher_cls_tokens = None
            print("Teacher logits loaded successfully!")
        except:
            print("Generating teacher soft labels...")
            # Quick baseline check - evaluate untrained student (only if not resuming)
            if not resume_from_checkpoint:
    print("Evaluating untrained student model...")
    temp_val_dataset = DistillationDataset(val_dataset, processor)
    temp_val_loader = DataLoader(temp_val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in temp_val_loader:
            if total >= 1000:  # Quick check on 1000 samples
                break
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = student_model(pixel_values)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    baseline_acc = 100 * correct / total
    print(f"Untrained student baseline accuracy: {baseline_acc:.2f}%")
    
    temp_train_dataset = DistillationDataset(train_dataset, processor)
    temp_train_loader = DataLoader(temp_train_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Generate teacher predictions with lower temperature for sharper targets
    teacher_logits, teacher_cls_tokens = generate_teacher_predictions(
        teacher_model, temp_train_loader, temperature=3.0  # Lower temperature for sharper targets
    )
            
            # Save teacher logits for future use
            torch.save(teacher_logits, teacher_logits_file)
            print(f"Teacher logits saved to {teacher_logits_file}")
    
    # Create final datasets with teacher predictions
    train_dataset_dist = DistillationDataset(train_dataset, processor, teacher_logits.numpy())
    val_dataset_dist = DistillationDataset(val_dataset, processor)
    
    # Create data loaders with smaller batch size for faster learning
    train_loader = DataLoader(train_dataset_dist, batch_size=128, shuffle=True, num_workers=0)  # Reduced from 512
    val_loader = DataLoader(val_dataset_dist, batch_size=256, shuffle=False, num_workers=0)
    
    # Setup optimizer with much higher learning rate
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=0.01)  # Increased from 3e-4
    total_steps = len(train_loader) * 60  # 60 epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps  # Reduced warmup
    )
    
    # Setup loss function with more aggressive settings
    loss_fn = DistillationLoss(alpha=0.7, temperature=2.0, use_cls_mse=False)  # Higher alpha, lower temp
    
    # Train with distillation
    trainer = DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=60
    )
    
    # Load checkpoint if resuming
    if resume_from_checkpoint:
        trainer.load_checkpoint(resume_from_checkpoint)
    
    print("Starting knowledge distillation training...")
    best_accuracy = trainer.train()
    print(f"Best validation accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    # To resume from checkpoint, change this to the checkpoint path
    resume_checkpoint = "checkpoint_epoch_50.pth"  # Set to None to start from scratch
    main(resume_from_checkpoint=resume_checkpoint)
