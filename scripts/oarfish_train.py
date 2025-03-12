import os
import sys
import glob
import logging
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from oarfish import data, classify, train, predict


def get_data_from_dir(base_dir, model):
    """Get paths and labels from directory structure"""
    paths = []
    labels = []
    class_counts = Counter()
    
    # Get data for each class
    for class_name in model.class_names:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            logging.warning(f"Directory not found: {class_dir}")
            continue
        
        # Get all npz files
        class_paths = glob.glob(os.path.join(class_dir, '*.npz'))
        class_label = model.class_mapping[class_name]
        
        paths.extend(class_paths)
        labels.extend([class_label] * len(class_paths))
        class_counts[class_name] = len(class_paths)
        
        logging.info(f"Found {len(class_paths)} images for class '{class_name}'")
    
    return paths, labels, class_counts


def analyze_class_balance(train_counts, val_counts, model):
    """Analyze and log class distribution"""
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    
    logging.info("  Class distribution:")
    logging.info("  Training set:")
    for class_name in model.class_names:
        count = train_counts[class_name]
        percentage = (count / total_train) * 100
        logging.info(f"   {class_name}: {count} images ({percentage:.1f}%)")
    
    logging.info("  Validation set:")
    for class_name in model.class_names:
        count = val_counts[class_name]
        percentage = (count / total_val) * 100
        logging.info(f"   {class_name}: {count} images ({percentage:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LWATV Image Classifier')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping patience')
    parser.add_argument('--dataset-dir', type=str, default='.',
                        help='Directory containing the binary and multi-class training/validation data')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save training logs')
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'two_stage_training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    if True:
        # Part 1 - Binary Classifier
        m = classify.BinaryLWATVClassifier()
        
        # Log training configuration
        logger.info("BINARY - Training Configuration:")
        for arg in vars(args):
            logger.info(f"BINARY - {arg}: {getattr(args, arg)}")
        
        # Get training and validation data
        train_paths, train_labels, train_counts = get_data_from_dir(f"{args.dataset_dir}/binary/train", m)
        val_paths, val_labels, val_counts = get_data_from_dir(f"{args.dataset_dir}binary/val", m)
        
        if not train_paths or not val_paths:
            logger.error("BINARY - No images found in one or both directories")
            sys.exit(1)
        
        # Analyze and log class distribution
        analyze_class_balance(train_counts, val_counts, m)
        
        logger.info(f"BINARY - Total training images: {len(train_paths)}")
        logger.info(f"BINARY - Total validation images: {len(val_paths)}")
        
        # Create datasets
        train_dataset = data.LWATVDataset(train_paths, train_labels)
        val_dataset = data.LWATVDataset(val_paths, val_labels)
        
        # Train the model
        model = train.train_model(
            classify.BinaryLWATVClassifier,
            train.ModelTrainer,
            train_dataset,
            val_dataset,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            patience=args.patience,
            checkpoint_dir=args.checkpoint_dir+'_binary'
        )
        
        # Run validation predictions on a few examples
        logger.info("BINARY - Example predictions on validation set:")
        n_examples = min(5, len(val_paths))
        for i in range(n_examples):
            prediction, confidence = predict.predict_image(
                model, 
                data.LWATVDataset([val_paths[i]])
            )
            true_label = m.class_names[val_labels[i]]
            logger.info(
                f"BINARY - Image: {os.path.basename(val_paths[i])} | "
                f"BINARY - True: {true_label} | "
                f"BINARY - Predicted: {prediction} | "
                f"BINARY - Confidence: {confidence:.2f}"
            )
        
    # Part 2 -  Multi Classifier
    m = classify.MultiLWATVClassifier()
    
    # Log training configuration
    logger.info("MULTI - Training Configuration:")
    for arg in vars(args):
        logger.info(f"MULTI - {arg}: {getattr(args, arg)}")
    
    # Get training and validation data
    train_paths, train_labels, train_counts = get_data_from_dir(f"{args.dataset_dir}/multi/train", m)
    val_paths, val_labels, val_counts = get_data_from_dir(f"{args.dataset_dir}/multi/val", m)
    
    if not train_paths or not val_paths:
        logger.error("MULTI - No images found in one or both directories")
        sys.exit(1)
    
    # Analyze and log class distribution
    analyze_class_balance(train_counts, val_counts, m)
    
    logger.info(f"MULTI - Total training images: {len(train_paths)}")
    logger.info(f"MULTI - Total validation images: {len(val_paths)}")
    
    # Create datasets
    train_dataset = data.LWATVDataset(train_paths, train_labels)
    val_dataset = data.LWATVDataset(val_paths, val_labels)
    
    # Train the model
    model = train.train_model(
        classify.MultiLWATVClassifier,
        train.ModelTrainer,
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir+'_multi'
    )
    
    # Run validation predictions on a few examples
    logger.info("MULTI - Example predictions on validation set:")
    n_examples = min(5, len(val_paths))
    for i in range(n_examples):
        prediction, confidence = predict.predict_image(
            model, 
            data.LWATVDataset([val_paths[i]])
        )
        true_label = m.class_names[val_labels[i]]
        logger.info(
            f"MULTI - Image: {os.path.basename(val_paths[i])} | "
            f"MULTI - True: {true_label} | "
            f"MULTI - Predicted: {prediction} | "
            f"MULTI - Confidence: {confidence:.2f}"
        )
    
    sys.exit(1)

if __name__ == "__main__":
    exit(main())
