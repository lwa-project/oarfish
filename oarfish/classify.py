import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLWATVClassifier(nn.Module):
    def __init__(self, num_classes=2, class_names=None):
        super().__init__()
        self.num_classes = num_classes
        if class_names is not None:
            if len(class_names) != self.num_classes:
                raise ValueError("'class_names' does not match the number of classes")
        else:
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        self.class_names = class_names
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 1. Source Analysis Branch (for Sun, Jupiter, and horizon regions)
        self.source_analyzer = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        
        # 2. Global Pattern Analysis Branch
        self.pattern_analyzer = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        
        # 3. Edge Detection Branch (for identifying sharp transitions)
        self.edge_detector = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        
        # Main feature processing
        self.main_network = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),  # Combines all branches
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))  # Reduced output size
        )
        
        # Feature dimension reduction
        self.feature_reducer = nn.Sequential(
            nn.Linear(256 * 16, 128),  # Reduce to a smaller attention dimension
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )
        
        # 1D horizon processing branch
        self.horizon_analyzer = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )
        
        # Add horizon feature processor to match dimensions with other features
        self.horizon_processor = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),  # Reduce to fixed length
            nn.Flatten(),
            nn.Linear(32 * 16, 128),   # Match attention dimension
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1)
        )
        
        # Enhanced astronomical feature processing
        self.astro_processor = nn.Sequential(
            nn.Linear(22, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 128),  # Match attention dimension
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        
        # Final classification
        self.final_classifier = nn.Sequential(
            nn.Linear(384, 128),  # Combined features (128 from attention + 128 from horizon + 128 from astro)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, image, horizon, astro_features):
        # Process each specialized branch
        source_features = self.source_analyzer(image)
        pattern_features = self.pattern_analyzer(image)
        edge_features = self.edge_detector(image)
        
        # Combine visual features
        combined = torch.cat([
            source_features,
            pattern_features,
            edge_features
        ], dim=1)
        
        # Process through main network
        x = self.main_network(combined)
        img_features = x.view(x.size(0), -1)  # Flatten
        
        # Reduce feature dimensions
        img_features = self.feature_reducer(img_features)
        
        # Process horizon features
        horizon_features = self.horizon_analyzer(horizon)
        horizon_features = self.horizon_processor(horizon_features)
        
        # Process astronomical features
        astro_processed = self.astro_processor(astro_features)
        
        # Prepare features for attention 
        # Stack all three feature types: [3, batch_size, embed_dim]
        features_stack = torch.stack([
            img_features,
            horizon_features,
            astro_processed
        ], dim=0)
        
        # Apply attention mechanism
        attended_features, _ = self.attention(
            features_stack,
            features_stack,
            features_stack
        )
        
        # Combine all attended features
        final_features = torch.cat([
            attended_features[0],  # image features
            attended_features[1],  # horizon features
            attended_features[2]   # astro features
        ], dim=1)
        
        return self.final_classifier(final_features)
    
    def get_class_name(self, class_idx):
        """Convert class index to class name"""
        return self.class_names[class_idx]
    
    def get_class_idx(self, class_name):
        """Convert class name to class index"""
        return self.class_mapping[class_name]


class BinaryLWATVClassifier(BaseLWATVClassifier):
    """
    Sub-class of BaseLWATVClassifier that knows about good and bad images.
    """
    
    def __init__(self):
        super().__init__(num_classes=2, class_names=['good', 'bad'])


class MultiLWATVClassifier(BaseLWATVClassifier):
    """
    Sub-class of BaseLWATVClassifier that knows about the following kinds of
    images:
     * good
     * medium_rfi
     * high_rfi
     * corrupted
     * sun
     * jupiter
    """
    def __init__(self):
        super().__init__(num_classes=6, class_names=['good', 'medium_rfi', 'high_rfi', 'corrupted', 'sun', 'jupiter'])
    
    
