# Impact Prediction Neural Network

## Overview

This neural network predicts the actual impact position and velocity achieved by the robot given desired target values. This is useful for:

1. **Trajectory Planning**: Understanding how the robot actually performs vs. desired targets
2. **Control Compensation**: Correcting for systematic errors in impact control
3. **Performance Analysis**: Analyzing the consistency and accuracy of the impact controller

## Architecture

### Input Features (3 dimensions):
- `px`: Desired x-position of impact (from p_f_des[0])
- `py`: Desired y-position of impact (from p_f_des[1]) 
- `vz`: Desired z-velocity at impact (from v_f_des[2])

Note: z-position is fixed at 0.1m (impact surface), and x,y velocities are always 0.

### Output Predictions (6 dimensions):
- `impact_pos_x`: Actual x-position at impact
- `impact_pos_y`: Actual y-position at impact
- `impact_pos_z`: Actual z-position at impact
- `impact_vel_x`: Actual x-velocity at impact
- `impact_vel_y`: Actual y-velocity at impact
- `impact_vel_z`: Actual z-velocity at impact

### Network Architecture:
```
Input (3) → FC(64) → BatchNorm → ReLU → Dropout(0.2) →
          → FC(128) → BatchNorm → ReLU → Dropout(0.2) →
          → FC(128) → BatchNorm → ReLU → Dropout(0.2) →
          → FC(64) → BatchNorm → ReLU → Dropout(0.2) →
          → FC(6) → Output
```

**Rationale for this architecture:**
- **4 hidden layers**: Sufficient depth to capture nonlinear mappings without overfitting
- **64-128-128-64 neurons**: Expanding then contracting pattern allows learning complex representations
- **BatchNorm**: Stabilizes training and improves convergence
- **Dropout (0.2)**: Prevents overfitting, especially important for smaller datasets
- **ReLU activation**: Simple, effective activation that avoids vanishing gradients

## Installation

```bash
pip install -r requirements-neural-network.txt
```

## Usage

### 1. Basic Training
```python
# The script will automatically find the latest dataset
python train_impact_prediction_network.py
```

### 2. Custom Configuration
```python
from train_impact_prediction_network import ImpactPredictor

# Custom model configuration
config = {
    'hidden_layers': [32, 64, 64, 32],  # Smaller network
    'dropout_rate': 0.1,                # Less dropout
    'learning_rate': 0.0005,            # Lower learning rate
    'batch_size': 16,                   # Smaller batches
    'num_epochs': 150,                  # Fewer epochs
}

predictor = ImpactPredictor(config)
features, targets = predictor.load_data_from_hdf5('path/to/your/data.hdf5')
# ... continue with training
```

### 3. Loading Trained Model
```python
import torch
import pickle
from train_impact_prediction_network import ImpactPredictionNetwork

# Load model
model = ImpactPredictionNetwork(hidden_layers=[64, 128, 128, 64])
model.load_state_dict(torch.load('impact_prediction_model.pth'))
model.eval()

# Load scalers
with open('impact_prediction_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
    
feature_scaler = scalers['feature_scaler']
target_scaler = scalers['target_scaler']

# Make predictions
new_features = [[0.55, 0.02, -0.3]]  # [px, py, vz]
features_norm = feature_scaler.transform(new_features)
features_tensor = torch.FloatTensor(features_norm)

with torch.no_grad():
    pred_norm = model(features_tensor).numpy()
    
predictions = target_scaler.inverse_transform(pred_norm)
print(f"Predicted impact_pos: {predictions[0][:3]}")
print(f"Predicted impact_vel: {predictions[0][3:]}")
```

## Output Files

After training, the script saves:

1. **`impact_prediction_model.pth`**: Trained PyTorch model weights
2. **`impact_prediction_scalers.pkl`**: Feature and target normalization scalers
3. **`model_config.json`**: Model architecture and training configuration
4. **`evaluation_metrics.json`**: Detailed performance metrics
5. **`impact_prediction_results.png`**: Comprehensive visualization plots

## Expected Performance

For a well-trained model, you should expect:

- **Position accuracy**: MAE < 5mm (0.005m) for each axis
- **Velocity accuracy**: MAE < 0.05 m/s for each axis  
- **Overall R² score**: > 0.9 for position, > 0.8 for velocity
- **Training time**: 2-5 minutes depending on dataset size

## Interpreting Results

### Good Performance Indicators:
- Training and validation loss curves converge without large gaps
- R² scores > 0.8 for all output components
- Prediction vs. true value scatter plots lie close to diagonal line
- Error distribution is centered around zero with small variance

### Potential Issues:
- **Overfitting**: Large gap between training and validation loss
- **Underfitting**: Both training and validation loss plateau at high values
- **Data quality**: Poor R² scores may indicate noisy or insufficient data
- **Architecture**: Very poor performance may require adjusting network size/depth

## Customization Tips

### For Smaller Datasets (< 500 samples):
```python
config = {
    'hidden_layers': [32, 64, 32],      # Smaller network
    'dropout_rate': 0.3,                # More dropout
    'learning_rate': 0.0005,            # Lower learning rate
}
```

### For Larger Datasets (> 2000 samples):
```python
config = {
    'hidden_layers': [128, 256, 256, 128, 64],  # Deeper network
    'dropout_rate': 0.1,                         # Less dropout
    'batch_size': 64,                            # Larger batches
}
```

### For Higher Precision Requirements:
```python
config = {
    'hidden_layers': [128, 256, 512, 256, 128], # Wider network
    'learning_rate': 0.0001,                     # Lower learning rate
    'num_epochs': 500,                           # More training
    'early_stopping_patience': 50,              # More patience
}
```
