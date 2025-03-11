import os
import logging
import numpy as np
from collections import Counter

from typing import List, Tuple, Optional, Type, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import KFold

from . import CODE_CHECKSUM, REPO_INFO
from .data import LWATVDataset
from .classify import BinaryLWATVClassifier, MultiLWATVClassifier

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model: nn.Module, num_epochs: int=10, num_steps: int=10,
                       device: str='cuda' if torch.cuda.is_available() else 'cpu',
                       tag: Optional[str]=None):
        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.device = device
        self.tag = tag or ''
        
        self.criterion = nn.CrossEntropyLoss()
        # Use a smaller learning rate and weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=2e-4,
            weight_decay=0.02,
            eps=1e-8
        )
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=6e-4,  # 3x base learning rate
            epochs=self.num_epochs,
            steps_per_epoch=self.num_steps,
            pct_start=0.25,  # Reach max_lr at 25% of training
            div_factor=10,  # Start at lr/10
            final_div_factor=100,  # End at lr/100
            anneal_strategy='cos'
        )
        
        # Initialize best metrics
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
    def save_checkpoint(self, epoch: int, val_acc: float, val_loss: float, val_cmatrix: torch.Tensor,
                              path: str='checkpoints'):
        """Save model checkpoint"""
        if self.tag:
            path += f"_{self.tag}"
            
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'code_checksum': CODE_CHECKSUM,
            'repo_info': REPO_INFO,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'val_cmatrix': val_cmatrix
        }
        torch.save(checkpoint, os.path.join(path, f'model_epoch_{epoch}.pt'))
        
        # Save best model separately
        if val_acc > self.best_val_acc:
            torch.save(checkpoint, os.path.join(path, 'best_model.pt'))
            print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
            
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        if 'code_checksum' in checkpoint:
            if checkpoint['code_checksum'] != CODE_CHECKSUM:
                print("Warning: checksum mis-match between software and checkpoint data")
        else:
            print("Warning: no checksum found in the checkpoint data")
        if 'repo_info' in checkpoint:
            if checkpoint['repo_info'] != REPO_INFO:
                print("Warning: git repo info mis-match between software and checkpoint data")
        else:
            print("Warning: no git repo info found in the checkpoint data")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, torch.Tensor]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        num_classes = self.model.num_classes
        confusion_matrix = torch.zeros(num_classes, num_classes)
        
        for inputs, horizon_features, astro_features, labels in train_loader:
            inputs = inputs.to(self.device)
            horizon_features = horizon_features.to(self.device)
            astro_features = astro_features.to(self.device)
            labels = labels.to(self.device)
 
            self.optimizer.zero_grad()
            outputs = self.model(inputs, horizon_features, astro_features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            # Clip gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Confusion matrix
            for t, p in zip(labels, predicted):
                confusion_matrix[t.item(), p.item()] += 1
                
        # Normalize by row
        confusion_matrix = 100. * confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True)
    
        return running_loss / len(train_loader), 100. * correct / total, confusion_matrix
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, torch.Tensor]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        num_classes = self.model.num_classes
        confusion_matrix = torch.zeros(num_classes, num_classes)
        
        with torch.no_grad():
            for inputs, horizon_features, astro_features, labels in val_loader:
                inputs = inputs.to(self.device)
                horizon_features = horizon_features.to(self.device)
                astro_features = astro_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs, horizon_features, astro_features)
                loss = self.criterion(outputs, labels) 
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Confusion matrix
                for t, p in zip(labels, predicted):
                    confusion_matrix[t.item(), p.item()] += 1
                    
        # Normalize by row
        confusion_matrix = 100. * confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True)

        return running_loss / len(val_loader), 100. * correct / total, confusion_matrix

    def get_uncertain_samples(self, dataloader: DataLoader, n_samples: int=10) -> List[Tuple[int, float, int]]:
        """Identify the most uncertain predictions"""
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                # Get predictions and confidences
                pred = probs.argmax(dim=1)
                if self.model.num_classes == 2:
                    # For binary: uncertainty = how close to 0.5
                    uncertainty = -torch.abs(probs[:, 1] - 0.5)
                    confidence = torch.max(probs[:, 1], 1 - probs[:, 1])
                else:
                    # For multiclass: uncertainty = 1 - (max_prob - second_max_prob)
                    top2_probs = torch.topk(probs, k=2, dim=1)[0]
                    uncertainty = -(top2_probs[:, 0] - top2_probs[:, 1])
                    confidence = top2_probs[:, 0]
                
                # Store results
                for j, (unc, conf, p) in enumerate(zip(uncertainty, confidence, pred)):
                    uncertainties.append((
                        i * dataloader.batch_size + j,  # Global index
                        conf.item(),                    # Confidence
                        p.item()                        # Predicted class
                    ))
        
        # Sort by uncertainty (highest first)
        uncertainties.sort(key=lambda x: x[1])  # Sort by confidence (ascending)
        
        # Return the n most uncertain samples
        return uncertainties[:n_samples]


def create_sampler(dataset: LWATVDataset) -> WeightedRandomSampler:
    """Create a weighted sampler to handle class imbalance"""
    targets = [label for _, _, label in dataset]
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


def create_balanced_sampler(dataset: LWATVDataset) -> WeightedRandomSampler:
    """
    Create a weighted sampler that ensures:
    1. 50/50 split between 'good' and all other classes combined
    2. At least one sample from each class in every batch
    3. Equal probability among non-'good' classes
    
    Args:
        dataset: Dataset containing (image, label) pairs
        
    Returns:
        WeightedRandomSampler instance
    """
   
    # Get all labels
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        labels = [label for _, _, label in dataset]
    
    # Count instances of each class
    class_counts = Counter(labels)
    
    # Calculate weights for each sample
    weights = []
    n_samples = len(labels)
    
    # Split target probabilities: 0.5 for 'good', 0.5 for others combined
    good_idx = dataset.model.get_class_idx('good') if hasattr(dataset, 'model') else 0
    n_classes = len(class_counts)
    
    # Calculate per-class target probabilities
    target_probs = {}
    for class_idx in range(n_classes):
        if class_idx == good_idx:
            target_probs[class_idx] = 0.5
        else:
            # Distribute remaining 0.5 probability equally among non-good classes
            target_probs[class_idx] = 0.5 / (n_classes - 1)
    
    # Calculate weights for each sample
    for label in labels:
        current_prob = class_counts[label] / n_samples
        target_prob = target_probs[label]
        weight = target_prob / current_prob
        weights.append(weight)
    
    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=n_samples,
        replacement=True
    )
    
    return sampler


def train_model(model: Type[nn.Module], model_trainer: Type[ModelTrainer],
                train_dataset: LWATVDataset, val_dataset: LWATVDataset, batch_size: int=32,
                num_epochs: int=10, patience: int=5, checkpoint_dir: str='checkpoints') -> nn.Module:
    """
    Train model with early stopping and checkpointing
    
    Args:
        patience: Number of epochs to wait for improvement before early stopping
    """
    # Create dataloaders
    train_sampler = create_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_sampler = create_balanced_sampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    
    # Initialize model and trainer
    model = model()
    trainer = model_trainer(model, num_epochs=num_epochs, num_steps=len(train_loader))
    
    # Training loop
    best_val_acc = 0
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc, train_cmatrix = trainer.train_epoch(train_loader)
        val_loss, val_acc, val_cmatrix = trainer.validate(val_loader)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
        # Log per-class accuracies
        for i, class_name in enumerate(model.class_names):
            logger.info(f'  {class_name} - Train: {train_cmatrix[i,i]:.2f}%,'
                        f'  Val: {val_cmatrix[i,i]:.2f}%')
       
        #if epoch % 5 == 0:
            #analyze_feature_importance(model, val_loader, trainer.device)
        
        # Save checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            trainer.save_checkpoint(epoch + 1, val_acc, val_loss, val_cmatrix, checkpoint_dir)
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            logger.info(f'Early stopping triggered! No improvement for {patience} epochs.')
            break
    
    # Load best model before returning
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        logger.info('Loading best model from checkpoints...')
        trainer.load_checkpoint(best_model_path)
    
    return model


class EnsembleTrainer:
    def __init__(self, model_class: Type[nn.Module], n_models: int = 5,
        device: str='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_class = model_class
        self.n_models = n_models
        self.device = device
        self.models: List[nn.Module] = []
        self.trainers: List[ModelTrainer] = []
        
    def train_with_bagging(self, train_dataset: LWATVDataset, val_dataset: LWATVDataset,
                                 batch_size: int=32, num_epochs: int=10,
                                 sample_fraction: float = 0.8, checkpoint_dir: str = 'ensemble_checkpoints',
                                patience: int = 5):
        """Train ensemble using bagging (bootstrap aggregating)"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        dataset_size = len(train_dataset)
        sample_size = int(dataset_size * sample_fraction)
        
        for i in range(self.n_models):
            logger.info(f"\nTraining model {i+1}/{self.n_models}")
            
            # Create bootstrap sample
            indices = np.random.choice(dataset_size, size=sample_size, replace=True)
            bootstrap_dataset = torch.utils.data.Subset(train_dataset, indices)
            
            # Initialize and train model
            model = self.model_class().to(self.device)
            trainer = ModelTrainer(model, device=self.device)
            
            train_loader = DataLoader(bootstrap_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Training loop with early stopping
            best_val_acc = 0
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            
            # Create model-specific checkpoint directory
            model_checkpoint_dir = os.path.join(checkpoint_dir, f'model_{i}')
            os.makedirs(model_checkpoint_dir, exist_ok=True)
            
            for epoch in range(num_epochs):
                train_loss, train_acc = trainer.train_epoch(train_loader)
                val_loss, val_acc = trainer.validate(val_loader)
                
                logger.info(f'Epoch {epoch+1}/{num_epochs}:')
                logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Save regular checkpoint
                trainer.save_checkpoint(
                    epoch + 1, 
                    val_acc, 
                    val_loss, 
                    os.path.join(model_checkpoint_dir, f'epoch_{epoch+1}.pt')
                )
                
                # Update best model if validation improves
                if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                    logger.info(f'Validation improved: acc {best_val_acc:.2f} -> {val_acc:.2f}')
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    trainer.save_checkpoint(
                        epoch + 1,
                        val_acc,
                        val_loss,
                        os.path.join(model_checkpoint_dir, 'best_model.pt')
                    )
                else:
                    epochs_without_improvement += 1
                
                # Early stopping check
                if epochs_without_improvement >= patience:
                    logger.info(f'Early stopping triggered after {patience} epochs without improvement')
                    break
            
            self.models.append(model)
            self.trainers.append(trainer)
    
    def train_with_kfold(self, train_dataset: LWATVDataset, val_dataset: Optional[LWATVDataset]=None,
                               batch_size: int=32, num_epochs: int=10,
                               checkpoint_dir: str='ensemble_checkpoints',
                               patience: int=5):
        """Train ensemble using k-fold cross validation"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        kfold = KFold(n_splits=self.n_models, shuffle=True)
        
        for i, (train_idx, fold_val_idx) in enumerate(kfold.split(train_dataset)):
            logger.info(f"\nTraining model {i+1}/{self.n_models}")
            
            # Create fold datasets
            fold_train = torch.utils.data.Subset(train_dataset, train_idx)
            fold_val = torch.utils.data.Subset(train_dataset, fold_val_idx)
            
            # Initialize and train model
            model = self.model_class().to(self.device)
            trainer = ModelTrainer(model, device=self.device)
            
            train_loader = DataLoader(fold_train, batch_size=batch_size, shuffle=True)
            fold_val_loader = DataLoader(fold_val, batch_size=batch_size)
            
            # Additional validation set if provided
            final_val_loader = None
            if val_dataset is not None:
                final_val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Training loop with early stopping
            best_val_acc = 0
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            
            # Create model-specific checkpoint directory
            model_checkpoint_dir = os.path.join(checkpoint_dir, f'model_{i}')
            os.makedirs(model_checkpoint_dir, exist_ok=True)
            
            for epoch in range(num_epochs):
                train_loss, train_acc = trainer.train_epoch(train_loader)
                val_loss, val_acc = trainer.validate(fold_val_loader)
                
                logger.info(f'Epoch {epoch+1}/{num_epochs}:')
                logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                logger.info(f'Fold Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # If we have a final validation set, use that for checkpointing
                if final_val_loader:
                    final_val_loss, final_val_acc = trainer.validate(final_val_loader)
                    logger.info(f'Final Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.2f}%')
                    val_loss = final_val_loss
                    val_acc = final_val_acc
                
                # Save regular checkpoint
                trainer.save_checkpoint(
                    epoch + 1,
                    val_acc,
                    val_loss,
                    os.path.join(model_checkpoint_dir, f'epoch_{epoch+1}.pt')
                )
                
                # Update best model if validation improves
                if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                    logger.info(f'Validation improved: acc {best_val_acc:.2f} -> {val_acc:.2f}')
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    trainer.save_checkpoint(
                        epoch + 1,
                        val_acc,
                        val_loss,
                        os.path.join(model_checkpoint_dir, 'best_model.pt')
                    )
                else:
                    epochs_without_improvement += 1
                
                # Early stopping check
                if epochs_without_improvement >= patience:
                    logger.info(f'Early stopping triggered after {patience} epochs without improvement')
                    break
            
            self.models.append(model)
            self.trainers.append(trainer)
    
    def predict(self, dataset: LWATVDataset) -> Tuple[int, float]:
        """Get ensemble prediction through majority voting"""
        predictions = []
        confidences = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                img_tensor, astro_tensor = dataset[0]
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                astro_tensor = astro_tensor.unsqueeze(0).to(self.device)
                
                output = model(img_tensor, astro_tensor)
                prob = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                conf = prob[0][pred].item()
                
                predictions.append(pred)
                confidences.append(conf)
        
        # Majority voting for final prediction
        final_pred = max(set(predictions), key=predictions.count)
        # Average confidence for the majority class
        maj_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == final_pred]
        final_conf = sum(maj_confidences) / len(maj_confidences)
        
        return final_pred, final_conf
    
    def load_ensemble(self, checkpoint_dir: str, epoch: Optional[int]=None):
        """Load a previously trained ensemble from checkpoints
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            epoch: Specific epoch to load. If None, loads the best model for each ensemble member
        """
        self.models = []
        self.trainers = []
        
        for i in range(self.n_models):
            model_checkpoint_dir = os.path.join(checkpoint_dir, f'model_{i}')
            
            if epoch is not None:
                model_path = os.path.join(model_checkpoint_dir, f'epoch_{epoch}.pt')
            else:
                model_path = os.path.join(model_checkpoint_dir, 'best_model.pt')
                
            if os.path.exists(model_path):
                model = self.model_class().to(self.device)
                trainer = ModelTrainer(model, device=self.device)
                loaded_epoch = trainer.load_checkpoint(model_path)
                logger.info(f"Loaded model {i} from epoch {loaded_epoch}")
                
                self.models.append(model)
                self.trainers.append(trainer)
            else:
                logger.warning(f"Checkpoint not found: {model_path}")
        
        if not self.models:
            raise ValueError("No model checkpoints found in directory")
            
    def save_ensemble_metadata(self, checkpoint_dir: str):
        """Save ensemble metadata including validation metrics"""
        metadata = {
            'n_models': self.n_models,
            'model_metrics': []
        }
        
        for i, trainer in enumerate(self.trainers):
            model_metadata = {
                'model_index': i,
                'best_val_acc': trainer.best_val_acc,
                'best_val_loss': trainer.best_val_loss,
                'best_epoch': trainer.best_epoch
            }
            metadata['model_metrics'].append(model_metadata)
        
        metadata_path = os.path.join(checkpoint_dir, 'ensemble_metadata.pt')
        torch.save(metadata, metadata_path)
        logger.info(f"Saved ensemble metadata to {metadata_path}")
        
    def load_ensemble_metadata(self, checkpoint_dir: str):
        """Load ensemble metadata"""
        metadata_path = os.path.join(checkpoint_dir, 'ensemble_metadata.pt')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            logger.info("Loaded ensemble metadata:")
            for model_meta in metadata['model_metrics']:
                logger.info(f"Model {model_meta['model_index']}: "
                          f"Best val acc: {model_meta['best_val_acc']:.2f}, "
                          f"Best epoch: {model_meta['best_epoch']}")


def analyze_checkpoint(checkpoint_path: str, val_dataset: Optional[LWATVDataset]=None) -> Dict[str, Any]:
    """
    Analyze a model checkpoint file to extract training metrics and optionally compute current metrics
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        val_dataset: Optional validation dataset to compute current metrics
        
    Returns:
        dict containing available metrics and model info
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Extract saved metrics
    metrics = {
        'epoch': checkpoint.get('epoch'),
        'saved_val_acc': checkpoint.get('val_acc'),
        'saved_val_loss': checkpoint.get('val_loss'),
        'saved_val_cmatrix': checkpoint.get('val_cmatrix')
    }
    
    # Try to determine model type from state dict
    state_dict = checkpoint['model_state_dict']
    if metrics['saved_val_cmatrix'].shape[0] == 2:
        model = BinaryLWATVClassifier()
    elif metrics['saved_val_cmatrix'].shape[0] == 6:
        model = MultiLWATVClassifier()
    else:
        raise RuntimeError("Unknown model")
    metrics['model_type'] = type(model).__name__
        
    # Load state dict into model
    model.load_state_dict(state_dict)
    
    # If validation dataset provided, compute current metrics
    if val_dataset is not None:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        correct = 0
        total = 0
        running_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, astro_features, labels in val_dataset:
                inputs = inputs.to(device)
                astro_features = astro_features.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs, astro_features)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        metrics['current_val_acc'] = 100. * correct / total
        metrics['current_val_loss'] = running_loss / len(val_dataset)
    
    return metrics

def print_checkpoint_analysis(checkpoint_path: str, val_dataset: Optional[LWATVDataset]=None):
    """Pretty print the checkpoint analysis results"""
    metrics = analyze_checkpoint(checkpoint_path, val_dataset)
    
    print(f"\nCheckpoint Analysis for: {checkpoint_path}")
    print(f"Model Type: {metrics['model_type']}")
    print(f"Saved at epoch: {metrics['epoch']}")
    
    if metrics['saved_val_acc'] is not None:
        print(f"\nSaved Metrics:")
        print(f"  Validation Accuracy: {metrics['saved_val_acc']:.2f}%")
        print(f"  Validation Loss: {metrics['saved_val_loss']:.4f}")
    
    if 'current_val_acc' in metrics:
        print(f"\nCurrent Metrics:")
        print(f"  Validation Accuracy: {metrics['current_val_acc']:.2f}%")
        print(f"  Validation Loss: {metrics['current_val_loss']:.4f}")
        
    if 'saved_val_cmatrix' in metrics:
        print(f"  Validation Confusion Matrix:")
        for row in metrics['saved_val_cmatrix']:
            print("   ", ' '.join([f"{v:6.2f}%" for v in row]))
