from pathlib import Path
import os
import time
from logging import Logger
from typing import Tuple, Dict, Any, Optional, Tuple, List

import torch
from torch.utils.data import DataLoader

from . import CODE_CHECKSUM
from .data import LWATVDataset
from .classify import *

def predict_image(model: BaseLWATVClassifier, dataset: LWATVDataset,
                  device: str='cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[int, float]:
    """Predict a single image"""
    model.eval()
    img_tensor, horizon_tensor, astro_tensor = dataset[0]  # Unpack the tuple
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    horizon_tensor = horizon_tensor.unsqueeze(0).to(device)
    astro_tensor = astro_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor, horizon_tensor, astro_tensor)  # Pass both tensors
        prob = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = prob[0][prediction].item()
    
    class_name = model.get_class_name(prediction)
    return prediction, confidence

def predict_with_uncertainty(model: BaseLWATVClassifier, dataset: LWATVDataset,
                             confidence_threshold: float=0.8, 
                             device: str='cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, Any]:
    """
    Predict a single image with uncertainty estimation
    
    Args:
        model: Trained model
        image_path: Path to image file
        confidence_threshold: Threshold for considering prediction as uncertain
        device: Computing device
    
    Returns:
        dict containing prediction, confidence, and uncertainty flag
    """
    model.eval()
    img_tensor, horizon_tensor, astro_tensor = dataset[0]
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    horizon_tensor = horizon_tensor.unsqueeze(0).to(device)
    astro_tensor = astro_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor, horizon_tensor, astro_tensor)
        probabilities = F.softmax(output, dim=1)
        
        # Get prediction and confidence
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        # Calculate entropy as uncertainty measure
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
        entropy = entropy.item()
        
        # Create class probabilities dictionary
        class_probs = {
            model.get_class_name(i): probabilities[0][i].item()
            for i in range(len(model.CLASS_NAMES))
        }
        
        result = {
            'prediction': model.get_class_name(prediction),
            'confidence': confidence,
            'entropy': entropy,
            'is_uncertain': confidence < confidence_threshold,
            'probabilities': class_probs
        }
        
        return result


try:
    from torch.serialization import add_safe_globals
    
    add_safe_globals({'oarfish.classify.BinaryLWATVClassifier': BinaryLWATVClassifier,
                      'oarfish.classify.MultiLWATVClassifier': MultiLWATVClassifier,
                     })
except ImportError:
    pass


class DualModelPredictor:
    """
    Class for batch processing OIMS files using both binary and multi-class models
    and computing a combined quality score
    """
    
    def __init__(self, binary_model_path: str, multi_model_path: str,
                       device: Optional[str]=None, logger: Optional[Logger]=None):
        """
        Initialize the dual model predictor
        
        Args:
            binary_model_path (str): Path to the trained binary model checkpoint
            multi_model_path (str): Path to the trained multi-class model checkpoint
            device (str, optional): Device to use for computation ('cuda' or 'cpu')
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.logger = logger
        
        self.load_time = time.time()
            
        # Load binary model
        self.binary_path = os.path.abspath(binary_model_path)
        self.binary_model = BinaryLWATVClassifier()
        binary_checkpoint = torch.load(binary_model_path, map_location=self.device,
                                       weights_only=False)
        self.binary_model.load_state_dict(binary_checkpoint['model_state_dict'])
        self.binary_model = self.binary_model.to(self.device)
        self.binary_model.eval()
        
        # Load multi-class model
        self.multi_path = os.path.abspath(multi_model_path)
        self.multi_model = MultiLWATVClassifier()
        multi_checkpoint = torch.load(multi_model_path, map_location=self.device,
                                      weights_only=False)
        self.multi_model.load_state_dict(multi_checkpoint['model_state_dict'])
        self.multi_model = self.multi_model.to(self.device)
        self.multi_model.eval()
        
        # Extract validation accuracies for reference
        self.binary_val_acc = binary_checkpoint.get('val_acc', 0.0)
        self.multi_val_acc = multi_checkpoint.get('val_acc', 0.0)
        
        if self.logger:
            self.logger.info(f"Loaded binary model with validation accuracy: {self.binary_val_acc:.2f}%")
            self.logger.info(f"Loaded multi model with validation accuracy: {self.multi_val_acc:.2f}%")
            
        self._batches_processed = 0
            
    def identify(self) -> Dict[str, Any]:
        ident = {'name': 'DualModelPredictor',
                 'binary_model': self.binary_path,
                 'binary_val_acc': self.binary_val_acc,
                 'multi_model': self.multi_path,
                 'multi_val_acc': self.multi_val_acc,
                 'load_time': self.load_time,
                 'device': self.device,
                 'code_checksum': CODE_CHECKSUM
                }
        return ident
        
    def compute_quality_score(self, binary_pred: int, binary_conf: float,
                                    multi_pred: int, multi_probs: torch.Tensor) -> Tuple[float, str]:
        """
        Compute quality score based on predictions from both models
        
        Args:
            binary_pred (int): Binary model prediction index
            binary_conf (float): Binary model confidence
            multi_pred (int): Multi-class model prediction index
            multi_probs (tensor): Probability distribution from multi-class model
            
        Returns:
            tuple: (quality score, final label)
        """
        # Get top 2 predictions from multi-class model
        values, indices = torch.topk(multi_probs, 2)
        first = (indices[0].item(), values[0].item())
        second = (indices[1].item(), values[1].item())
        
        # Compute quality score as specified
        q = 0.0
        if self.binary_model.get_class_name(binary_pred) == 'good':
            q += binary_conf * 0.25
        else:
            q += (1 - binary_conf) * 0.25
            
        if self.multi_model.get_class_name(first[0]) == 'good':
            q += first[1] + (1 - second[1])
        elif self.multi_model.get_class_name(second[0]) == 'good':
            q += second[1] + (1 - first[1])
            
        q /= 2.25
        
        # Determine final label based on quality score
        if q > 0.75:
            final = 'good'
        elif q > 0.5:
            final = 'low_rfi'
        elif q > 0.3:
            final = 'medium_rfi'
        else:
            final = self.multi_model.get_class_name(first[0])
            
        return q, final
    
    def empty_cache(self):
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            
    def predict_dataset(self, dataset: LWATVDataset, batch_size: Optional[int]=None) -> List[Dict[str, Any]]:
        results = []
        
        if batch_size is None:
            batch_size = min(len(dataset), 32)
            
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        
        # Process batches
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                self._batches_processed += 1
                if self._batches_processed % 10 == 0:
                    self.empty_cache()
                    
                try:
                    img_tensors, hrz_tensors, astro_tensors, info_batch = batch_data
                except ValueError:
                    img_tensors, hrz_tensors, astro_tensors = batch_data
                    info_batch = {}
                    
                # Move to device
                img_tensors = img_tensors.to(self.device)
                hrz_tensors = hrz_tensors.to(self.device)
                astro_tensors = astro_tensors.to(self.device)
                
                # Get binary model predictions
                binary_outputs = self.binary_model(img_tensors, hrz_tensors, astro_tensors)
                binary_probs = torch.softmax(binary_outputs, dim=1)
                binary_preds = binary_outputs.argmax(dim=1)
                binary_confs = torch.gather(binary_probs, 1, binary_preds.unsqueeze(1))
                
                # Get multi-class model predictions
                multi_outputs = self.multi_model(img_tensors, hrz_tensors, astro_tensors)
                multi_probs = torch.softmax(multi_outputs, dim=1)
                multi_preds = multi_outputs.argmax(dim=1)
                multi_confs = torch.gather(multi_probs, 1, multi_preds.unsqueeze(1))
                
                # Process each item in the batch
                for i in range(len(binary_preds)):
                    info = {}
                    for key, value in info_batch.items():
                        if isinstance(value, torch.Tensor) and value.numel() > i:
                            info[key] = value[i].item()
                        else:
                            info[key] = value
                    
                    # Extract predictions
                    binary_pred = binary_preds[i].item()
                    binary_conf = binary_confs[i].item()
                    multi_pred = multi_preds[i].item()
                    multi_conf = multi_confs[i].item()
                    
                    # Compute quality score
                    quality_score, final_label = self.compute_quality_score(
                        binary_pred, binary_conf, multi_pred, multi_probs[i]
                    )
                    
                    # Get top 2 multi-class predictions
                    values, indices = torch.topk(multi_probs[i], 2)
                    top2 = [
                        (self.multi_model.get_class_name(indices[0].item()), values[0].item()),
                        (self.multi_model.get_class_name(indices[1].item()), values[1].item())
                    ]
                    
                    # Create probability dictionaries
                    binary_class_probs = {
                        self.binary_model.get_class_name(j): binary_probs[i][j].item() 
                        for j in range(len(self.binary_model.class_names))
                    }
                    
                    multi_class_probs = {
                        self.multi_model.get_class_name(j): multi_probs[i][j].item() 
                        for j in range(len(self.multi_model.class_names))
                    }
                    
                    # Create result entry
                    result = {
                        'start_time': info.get('start_time', 0),
                        'center_freq': info.get('start_freq', 0),
                        'binary_prediction': self.binary_model.get_class_name(binary_pred),
                        'binary_confidence': binary_conf,
                        'multi_prediction': self.multi_model.get_class_name(multi_pred),
                        'multi_confidence': multi_conf,
                        'multi_top2': top2,
                        'quality_score': quality_score,
                        'final_label': final_label
                    }
                    
                    results.append(result)
        
        return results
