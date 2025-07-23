#!/usr/bin/env python3
"""
PyTorch Neural Network for Impact Prediction

This script trains a fully connected neural network to predict impact_pos and impact_vel
from desired final position (p_f_des) and velocity (v_f_des) using data collected
from the impact control pipeline.

Input features (3):
    - p_f_des[0]: x position (px)
    - p_f_des[1]: y position (py) 
    - v_f_des[2]: z velocity (vz)

Output features (6):
    - impact_pos[0]: actual x position
    - impact_pos[1]: actual y position
    - impact_pos[2]: actual z position
    - impact_vel[0]: actual x velocity
    - impact_vel[1]: actual y velocity
    - impact_vel[2]: actual z velocity
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import datetime
import json
import pickle


class ImpactDataset(Dataset):
    """Custom PyTorch Dataset for impact prediction data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            features: Input features (N, 3) - [px, py, vz]
            targets: Target outputs (N, 6) - [impact_pos (3), impact_vel (3)]
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class ImpactPredictionNetwork(nn.Module):
    """Fully connected neural network for impact prediction"""
    
    def __init__(self, input_dim: int = 3, output_dim: int = 6, 
                 hidden_layers: List[int] = [64, 128, 128, 64], 
                 dropout_rate: float = 0.2):
        """
        Args:
            input_dim: Number of input features (3: px, py, vz)
            output_dim: Number of output features (6: impact_pos + impact_vel)
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(ImpactPredictionNetwork, self).__init__()
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class ImpactPredictor:
    """Main class for training and evaluating the impact prediction model"""
    
    def __init__(self, model_config: Dict = None):
        """
        Args:
            model_config: Dictionary with model configuration
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Default model configuration
        default_config = {
            'hidden_layers': [64, 128, 128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 200,
            'early_stopping_patience': 20,
            'weight_decay': 1e-5
        }
        
        self.config = default_config if model_config is None else {**default_config, **model_config}
        
        # Initialize components
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        self.train_losses = []
        self.val_losses = []
        
    def load_data_from_hdf5(self, hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from HDF5 file collected by impact control pipeline
        
        Args:
            hdf5_path: Path to the HDF5 file
            
        Returns:
            features: Input features (N, 3) - [px, py, vz]
            targets: Target outputs (N, 6) - [impact_pos (3), impact_vel (3)]
        """
        features = []
        targets = []
        
        print(f"Loading data from: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Find all run groups
            run_keys = [key for key in f.keys() if key.startswith('run_')]
            print(f"Found {len(run_keys)} runs in the dataset")
            
            valid_samples = 0
            for run_key in run_keys:
                try:
                    run_group = f[run_key]
                    
                    # Extract desired targets
                    if 'p_f_des' in run_group and 'v_f_des' in run_group:
                        p_f_des = run_group['p_f_des'][()]
                        v_f_des = run_group['v_f_des'][()]
                        
                        # Extract actual impact data
                        if 'impact_pos' in run_group and 'impact_vel' in run_group:
                            impact_pos = run_group['impact_pos'][()]
                            impact_vel = run_group['impact_vel'][()]
                            
                            # Check for valid data (not None/NaN)
                            if (impact_pos is not None and impact_vel is not None and
                                not np.any(np.isnan(impact_pos)) and not np.any(np.isnan(impact_vel))):
                                
                                # Input features: [px, py, vz]
                                feature = np.array([p_f_des[0], p_f_des[1], v_f_des[2]])
                                
                                # Output targets: [impact_pos (3), impact_vel (3)]
                                target = np.concatenate([impact_pos, impact_vel])
                                
                                features.append(feature)
                                targets.append(target)
                                valid_samples += 1
                    
                except Exception as e:
                    print(f"Error processing {run_key}: {e}")
                    continue
        
        if valid_samples == 0:
            raise ValueError("No valid samples found in the dataset!")
        
        features = np.array(features)
        targets = np.array(targets)
        
        print(f"Loaded {valid_samples} valid samples")
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Feature ranges:")
        print(f"  px: [{features[:, 0].min():.3f}, {features[:, 0].max():.3f}]")
        print(f"  py: [{features[:, 1].min():.3f}, {features[:, 1].max():.3f}]")
        print(f"  vz: [{features[:, 2].min():.3f}, {features[:, 2].max():.3f}]")
        
        return features, targets
    
    def prepare_data(self, features: np.ndarray, targets: np.ndarray, 
                    test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Prepare and normalize the data for training
        
        Args:
            features: Input features
            targets: Target outputs
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            
        Returns:
            Tuple of (train_loader, val_loader, test_features, test_targets)
        """
        # Split data into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=42
        )
        
        # Split train+val into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        print(f"Data split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Normalize features and targets
        X_train_norm = self.scaler_features.fit_transform(X_train)
        X_val_norm = self.scaler_features.transform(X_val)
        X_test_norm = self.scaler_features.transform(X_test)
        
        y_train_norm = self.scaler_targets.fit_transform(y_train)
        y_val_norm = self.scaler_targets.transform(y_val)
        y_test_norm = self.scaler_targets.transform(y_test)
        
        # Create datasets
        train_dataset = ImpactDataset(X_train_norm, y_train_norm)
        val_dataset = ImpactDataset(X_val_norm, y_val_norm)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'], 
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'], 
            shuffle=False, num_workers=0
        )
        
        return train_loader, val_loader, X_test_norm, y_test_norm
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the neural network"""
        
        # Initialize model
        self.model = ImpactPredictionNetwork(
            hidden_layers=self.config['hidden_layers'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        print(f"Model architecture: {self.config['hidden_layers']}")
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_impact_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["num_epochs"]}], '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_impact_model.pth'))
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the trained model on test data"""
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Predict
        with torch.no_grad():
            y_pred_norm = self.model(X_test_tensor).cpu().numpy()
        
        # Denormalize predictions and targets
        y_pred = self.scaler_targets.inverse_transform(y_pred_norm)
        y_true = self.scaler_targets.inverse_transform(y_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Per-output metrics
        output_names = ['impact_pos_x', 'impact_pos_y', 'impact_pos_z', 
                       'impact_vel_x', 'impact_vel_y', 'impact_vel_z']
        
        per_output_metrics = {}
        for i, name in enumerate(output_names):
            per_output_metrics[name] = {
                'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'r2': r2_score(y_true[:, i], y_pred[:, i])
            }
        
        # Position and velocity specific metrics
        pos_mse = mean_squared_error(y_true[:, :3], y_pred[:, :3])
        vel_mse = mean_squared_error(y_true[:, 3:], y_pred[:, 3:])
        pos_mae = mean_absolute_error(y_true[:, :3], y_pred[:, :3])
        vel_mae = mean_absolute_error(y_true[:, 3:], y_pred[:, 3:])
        
        metrics = {
            'overall': {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            },
            'position': {
                'mse': pos_mse,
                'mae': pos_mae,
                'rmse': np.sqrt(pos_mse)
            },
            'velocity': {
                'mse': vel_mse,
                'mae': vel_mae,
                'rmse': np.sqrt(vel_mse)
            },
            'per_output': per_output_metrics,
            'predictions': y_pred,
            'true_values': y_true
        }
        
        return metrics
    
    def plot_results(self, metrics: Dict, save_dir: str = './'):
        """Plot training results and evaluation metrics"""
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training curves
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Prediction vs True scatter plots
        y_pred = metrics['predictions']
        y_true = metrics['true_values']
        
        # Position scatter plot
        axes[0, 1].scatter(y_true[:, :3].flatten(), y_pred[:, :3].flatten(), alpha=0.6)
        axes[0, 1].plot([y_true[:, :3].min(), y_true[:, :3].max()], 
                       [y_true[:, :3].min(), y_true[:, :3].max()], 'r--')
        axes[0, 1].set_xlabel('True Impact Position')
        axes[0, 1].set_ylabel('Predicted Impact Position')
        axes[0, 1].set_title(f'Position Prediction (R² = {r2_score(y_true[:, :3].flatten(), y_pred[:, :3].flatten()):.3f})')
        axes[0, 1].grid(True)
        
        # Velocity scatter plot
        axes[0, 2].scatter(y_true[:, 3:].flatten(), y_pred[:, 3:].flatten(), alpha=0.6)
        axes[0, 2].plot([y_true[:, 3:].min(), y_true[:, 3:].max()], 
                       [y_true[:, 3:].min(), y_true[:, 3:].max()], 'r--')
        axes[0, 2].set_xlabel('True Impact Velocity')
        axes[0, 2].set_ylabel('Predicted Impact Velocity')
        axes[0, 2].set_title(f'Velocity Prediction (R² = {r2_score(y_true[:, 3:].flatten(), y_pred[:, 3:].flatten()):.3f})')
        axes[0, 2].grid(True)
        
        # Per-output error bars
        output_names = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
        mae_values = [metrics['per_output'][f'impact_{name}']['mae'] for name in output_names]
        
        axes[1, 0].bar(output_names, mae_values)
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('MAE by Output Component')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True)
        
        # Error distribution histogram
        errors = y_pred - y_true
        axes[1, 1].hist(errors.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True)
        
        # R² scores by component
        r2_values = [metrics['per_output'][f'impact_{name}']['r2'] for name in output_names]
        colors = ['blue' if r2 > 0.8 else 'orange' if r2 > 0.6 else 'red' for r2 in r2_values]
        
        axes[1, 2].bar(output_names, r2_values, color=colors)
        axes[1, 2].set_ylabel('R² Score')
        axes[1, 2].set_title('R² Score by Output Component')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True)
        axes[1, 2].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good (0.8)')
        axes[1, 2].axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Fair (0.6)')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'impact_prediction_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_scalers(self, save_dir: str = './'):
        """Save the trained model and scalers"""
        
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, 'impact_prediction_model.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # Save scalers
        scalers = {
            'feature_scaler': self.scaler_features,
            'target_scaler': self.scaler_targets
        }
        
        scaler_path = os.path.join(save_dir, 'impact_prediction_scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        
        # Save configuration
        config_path = os.path.join(save_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Scalers saved to: {scaler_path}")
        print(f"Config saved to: {config_path}")


def find_latest_dataset():
    """Find the latest hardware dataset"""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    if not os.path.exists(results_dir):
        return None
    
    # Find hardware data directories
    hw_dirs = [d for d in os.listdir(results_dir) if d.startswith('hardware_data_')]
    if not hw_dirs:
        return None
    
    hw_dirs.sort(reverse=True)  # Most recent first
    
    for dirname in hw_dirs:
        hdf5_path = os.path.join(results_dir, dirname, 'hardware_collected_dataset_gen3.hdf5')
        if os.path.exists(hdf5_path):
            return hdf5_path
    
    return None


def main():
    """Main training script"""
    print("=" * 60)
    print("IMPACT PREDICTION NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Find dataset
    dataset_path = find_latest_dataset()
    if dataset_path is None:
        print("No hardware dataset found. Please run data collection first.")
        return
    
    print(f"Using dataset: {dataset_path}")
    
    # Custom model configuration (you can modify these)
    model_config = {
        'hidden_layers': [64, 128, 128, 64],  # 4-layer network, suitable for this task
        'dropout_rate': 0.2,                  # Moderate dropout for regularization
        'learning_rate': 0.001,               # Adam default
        'batch_size': 32,                     # Good balance for small-medium datasets
        'num_epochs': 300,                    # Allow for longer training
        'early_stopping_patience': 25,       # Stop if no improvement for 25 epochs
        'weight_decay': 1e-5                  # L2 regularization
    }
    
    # Initialize predictor
    predictor = ImpactPredictor(model_config)
    
    try:
        # Load data
        features, targets = predictor.load_data_from_hdf5(dataset_path)
        
        # Prepare data
        train_loader, val_loader, X_test, y_test = predictor.prepare_data(features, targets)
        
        # Train model
        predictor.train_model(train_loader, val_loader)
        
        # Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test)
        
        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Overall MSE: {metrics['overall']['mse']:.6f}")
        print(f"Overall MAE: {metrics['overall']['mae']:.6f}")
        print(f"Overall R²: {metrics['overall']['r2']:.4f}")
        print(f"Overall RMSE: {metrics['overall']['rmse']:.6f}")
        print()
        print(f"Position MSE: {metrics['position']['mse']:.6f}")
        print(f"Position MAE: {metrics['position']['mae']:.6f}")
        print(f"Position RMSE: {metrics['position']['rmse']:.6f}")
        print()
        print(f"Velocity MSE: {metrics['velocity']['mse']:.6f}")
        print(f"Velocity MAE: {metrics['velocity']['mae']:.6f}")
        print(f"Velocity RMSE: {metrics['velocity']['rmse']:.6f}")
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(dataset_path), 'neural_network_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and results
        predictor.save_model_and_scalers(output_dir)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        # Convert numpy arrays to lists for JSON serialization
        metrics_to_save = {
            'overall': metrics['overall'],
            'position': metrics['position'],
            'velocity': metrics['velocity'],
            'per_output': metrics['per_output']
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Plot results
        predictor.plot_results(metrics, output_dir)
        
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
