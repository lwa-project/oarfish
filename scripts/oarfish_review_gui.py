import os
import random
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from datetime import datetime
import logging
from functools import lru_cache
import argparse

from oarfish.models import *
from oarfish.classify import BinaryLWATVClassifier, MultiLWATVClassifier
from oarfish.data import LWATVDataset, load_lwatv_data, info_to_wcs
from oarfish.utils import extract_sky, extract_horizon, extract_beyond_horizon, characterize_sky, characterize_horizon
from oarfish.predict import DualModelPredictor


DEFAULT_BINARY = get_default_binary_model()
DEFAULT_MULTI = get_default_multi_model()


class ClassificationReviewApp:
    def __init__(self, root, dataset_dir, review_dir, binary_model, multi_model):
        self.root = root
        self.root.title("LWATV Classification Review")
        
        # Configure paths
        self.base_path = dataset_dir
        self.output_base = review_dir
        self.setup_directories()
        
        # Load models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = DualModelPredictor(binary_model, multi_model, device=self.device)
        
        # Initialize state
        self.current_file = None
        self.current_image_info = None
        self.current_predictions = None
        self.stats = {"reviewed": 0, "changed": 0}
        self.history = []
        self.history_pos = -1
        
        # Setup logging
        self.setup_logging()
        
        # Create GUI
        self.setup_gui()
        
        # Bind keyboard shortcuts
        self.setup_shortcuts()
        
        # Load first image
        self.load_random_image()
    
    def prev_image(self):
        """Load previous image from history"""
        if self.history_pos > 0:
            self.history_pos -= 1
            image_path = self.history[self.history_pos]
            self.load_image(image_path, add_to_history=False)
        else:
            pass
    
    def next_image(self):
        """Load next image"""
        if self.history_pos < len(self.history) - 1:
            # Navigate forward in history
            self.history_pos += 1
            image_path = self.history[self.history_pos]
            self.load_image(image_path, add_to_history=False)
        else:
            # At the end of history, load a new random image
            self.stats["reviewed"] += 1
            self.update_stats()
            self.load_random_image()
    
    def update_stats(self):
        """Update the statistics display"""
        self.stats_label.config(
            text=f"Reviewed: {self.stats['reviewed']} | Changed: {self.stats['changed']}"
        )
    
    def setup_directories(self):
        """Create output directories"""
        os.makedirs(self.output_base, exist_ok=True)
        for class_name in ['good', 'medium_rfi', 'high_rfi', 'corrupted', 'sun', 'jupiter']:
            os.makedirs(os.path.join(self.output_base, class_name), exist_ok=True)
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            filename=os.path.join(self.output_base, 'review_log.txt'),
            level=logging.INFO,
            format='%(asctime)s,%(message)s'
        )
    
    def setup_gui(self):
        """Create the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Image Information", padding="5")
        info_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # File info
        self.file_label = ttk.Label(info_frame, text="")
        self.file_label.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Current classification
        self.class_label = ttk.Label(info_frame, text="")
        self.class_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # Model predictions frame
        pred_frame = ttk.LabelFrame(main_frame, text="Model Predictions", padding="5")
        pred_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Binary model predictions
        self.binary_label = ttk.Label(pred_frame, text="")
        self.binary_label.grid(row=0, column=0, sticky=tk.W)
        
        # Multi-class model predictions
        self.multi_label = ttk.Label(pred_frame, text="")
        self.multi_label.grid(row=1, column=0, sticky=tk.W)
        
        # Quality prediction
        self.quality_label = ttk.Label(pred_frame, text="")
        self.quality_label.grid(row=2, column=0, sticky=tk.W)
        
        # Final verdict
        self.final_label = ttk.Label(pred_frame, text="")
        self.final_label.grid(row=3, column=0, sticky=tk.W)
        
        # Image metrics frame
        metrics_frame = ttk.LabelFrame(main_frame, text="Image Metrics", padding="5")
        metrics_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Metrics display
        self.metrics_label = ttk.Label(metrics_frame, text="")
        self.metrics_label.grid(row=0, column=0, sticky=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Previous (Left)", command=self.prev_image).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Next (Right)", command=self.next_image).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Change Class (c)", command=self.change_class).grid(row=0, column=2, padx=5)
        
        # Stats display
        self.stats_label = ttk.Label(main_frame, text="Reviewed: 0 | Changed: 0")
        self.stats_label.grid(row=5, column=0, columnspan=3)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('c', lambda e: self.change_class())
    
    def get_model_predictions(self, image_path):
        """Get predictions from both models"""
        dataset = LWATVDataset([image_path])
        return self.predictor.predict_dataset(dataset)
    
    def get_image_metrics(self, image_path):
        """Extract relevant metrics from the image"""
        si, sv, info = load_lwatv_data(image_path)
        wcs, topo_wcs = info_to_wcs(info, si.shape[-1])
        
        sky_i, sky_v = extract_sky(si, sv, topo_wcs)
        hrz_i, hrz_v = extract_horizon(si, sv, topo_wcs)
        byd_i, byd_v = extract_beyond_horizon(si, sv, topo_wcs)
        
        sky_metrics = characterize_sky(sky_i, sky_v)
        hrz_metrics = characterize_horizon(hrz_i, hrz_v)
        
        return {
            'sky': sky_metrics,
            'horizon': hrz_metrics,
            'info': info
        }
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _load_image_paths(base_path):
        image_paths = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.npz'):
                    image_paths.append(os.path.join(root, file))
        return image_paths
        
    def load_random_image(self):
        """Load a random image from the dataset"""
        # Get all image paths
        image_paths = self._load_image_paths(self.base_path)
        
        # Select random image
        image_path = random.choice(image_paths)
        self.load_image(image_path)
    
    def load_image(self, image_path, add_to_history=True):
        """Load and display an image"""
        try:
            # Load image data
            si, sv, info = load_lwatv_data(image_path)
            
            # Get predictions
            predictions = self.get_model_predictions(image_path)
            
            # Get metrics
            metrics = self.get_image_metrics(image_path)
            
            # Store current state
            self.current_file = image_path
            self.current_image_info = info
            self.current_predictions = predictions
            
            # Update the history
            if add_to_history:
                # Trim history to current position
                if self.history_pos < len(self.history) - 1:
                    self.history = self.history[:self.history_pos + 1]
                    
                # Add new image to history
                self.history.append(image_path)
                self.history_pos = len(self.history) - 1
                
                # Limit history size
                if len(self.history) > 100:
                    self.history = self.history[-100:]
                    self.history_pos = len(self.history) - 1
                    
            # Update display
            self.display_image(si, sv)
            self.update_info_display(image_path, info)
            self.update_prediction_display(predictions)
            self.update_metrics_display(metrics)
            
        except Exception as e:
            import traceback
            logging.error(f"Error loading {image_path}: {str(e)}")
            logging.error(f"Trackback: {traceback.format_exc()}")
            self.load_random_image()
    
    def display_image(self, si, sv):
        """Display the Stokes I and V images"""
        # Normalize images
        vmin, vmax = 0, np.percentile(si, 99.75)
        vmax = max(vmax, 1e-8)
        si = np.clip(si, vmin, vmax) / vmax
        sv = np.clip(sv, vmin, vmax) / vmax
        
        # Convert to PIL images
        si_img = Image.fromarray((si[::-1,:] * 255).astype(np.uint8))
        sv_img = Image.fromarray((sv[::-1,:] * 255).astype(np.uint8))
        
        # Combine images side by side
        combined = Image.new('L', (si_img.width * 2, si_img.height))
        combined.paste(si_img, (0, 0))
        combined.paste(sv_img, (si_img.width, 0))
        
        # Resize if needed
        max_size = 512
        ratio = max_size / combined.width
        new_size = (max_size, int(combined.height * ratio))
        combined = combined.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(combined)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
    
    def update_info_display(self, image_path, info):
        """Update the information display"""
        self.file_label.config(text=f"File: {os.path.basename(image_path)}")
        self.class_label.config(text=f"Current Class: {self.get_current_class(image_path)}")
    
    def update_prediction_display(self, predictions):
        """Update the model predictions display"""
        
        predictions = predictions[0]
        binary_pred = predictions['binary_prediction']
        binary_conf = predictions['binary_confidence']
        multi_pred  = predictions['multi_prediction']
        multi_conf  = predictions['multi_confidence']
        multi_top_2 = predictions['multi_top2']
        q     = predictions['quality_score']
        final = predictions['final_label']
        
        binary_text = f"Binary: {binary_pred} ({binary_conf:.2%})"
        self.binary_label.config(text=binary_text)
        
        multi_text = "Multi-class:\n"
        for label, conf in multi_top_2:
            multi_text += f"  {label}: {conf:.2%}\n"
        self.multi_label.config(text=multi_text[:-1])
        
        self.quality_label.config(text=f"Quality Score: {q:.2%}")
        self.final_label.config(text=f"Final Label: {final}")
    
    def update_metrics_display(self, metrics):
        """Update the metrics display"""
        sky = metrics['sky']
        hrz = metrics['horizon']
        info = metrics['info']
        
        text = f"Frequency: {info['start_freq']/1e6:.1f} MHz\n"
        text += f"Sky RMS (I/V): {sky['rms_i']:.3f}/{sky['rms_v']:.3f}\n"
        text += f"Horizon Max (I/V): {hrz['max_i']:.3f}/{hrz['max_v']:.3f}\n"
        text += f"High Fraction (I/V): {hrz['frac_high_i']:.3f}/{hrz['frac_high_v']:.3f}"
        
        self.metrics_label.config(text=text)
    
    def get_current_class(self, image_path):
        """Determine current class based on file location"""
        for class_name in ['good', 'medium_rfi', 'high_rfi', 'corrupted', 'sun', 'jupiter']:
            if class_name in image_path:
                return class_name
        return "unknown"
    
    def change_class(self):
        """Open dialog to change image class"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Change Class")
        
        # Create buttons for each class
        classes = ['good', 'medium_rfi', 'high_rfi', 'corrupted', 'sun', 'jupiter']
        for i, class_name in enumerate(classes):
            ttk.Button(
                dialog,
                text=class_name,
                command=lambda c=class_name: self.apply_class_change(c, dialog)
            ).grid(row=i//2, column=i%2, padx=5, pady=5)
    
    def apply_class_change(self, new_class, dialog):
        """Apply the class change"""
        if self.current_file:
            old_class = self.get_current_class(self.current_file)
            if old_class != new_class:
                # Create new filename
                old_name = os.path.basename(self.current_file)
                new_path = os.path.join(self.output_base, new_class, old_name)
                
                # Save the file
                np.savez(
                    new_path,
                    data=self.current_image_info['data'],
                    info=self.current_image_info
                )
                
                # Log the change
                logging.info(f"Changed class: {old_name} from {old_class} to {new_class}")
                
                # Update stats
                self.stats["changed"] += 1
                self.update_stats()
        
        dialog.destroy()
        self.load_random_image()


def main():
    parser = argparse.ArgumentParser(
        description='GUI to help understand how the training/validation datasets were classified',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset-dir', type=str, default='.',
                        help='directory containing training/validation data to review')
    parser.add_argument('--binary-model', type=str, default=DEFAULT_BINARY,
                        help='binary model to use for prediction')
    parser.add_argument('--multi-model', type=str, default=DEFAULT_MULTI,
                        help='multi-class model to use for prediction')
    parser.add_argument('--review-dir', type=str, default='review',
                        help='directory to write review reports to')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ClassificationReviewApp(root,
                                  args.dataset_dir, args.review_dir,
                                  args.binary_model, args.multi_model)
    root.mainloop()


if __name__ == "__main__":
    main()
