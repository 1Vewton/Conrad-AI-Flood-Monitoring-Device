# Flood Prediction-Backend

The backend of the system, carrying out the function of storing and fetching data, prediction and device management. 

# Components

Different modules that form the whole system. 

## Predictor

A hybrid machine learning model for time series forecasting that combines an ensemble of LightGBM models with Linear Regression for robust predictions. This module is responsible for predicting the future water level based on data fetched from the database. 

### Overview

This implementation provides a flexible time series forecasting solution that leverages both tree-based models and traditional linear regression. The hybrid approach automatically weights the contributions of each model type based on validation performance, making it adaptable to various time series patterns.

### Key Features

- **Hybrid Architecture**: Combines LightGBM ensemble with Linear Regression
- **Automatic Weighting**: Dynamically adjusts model contributions based on validation error
- **Asynchronous Training**: Parallel model training for improved performance
- **Comprehensive Feature Engineering**: Extracts meaningful patterns from time series windows
- **Rolling Forecast**: Multi-step ahead prediction capability

### Model Components

#### 1. Feature Engineering (`create_features` function)
Extracts both local and global patterns from time series windows:
- **Window Statistics**: First, middle, and last values, mean, standard deviation, delta
- **Trend Analysis**: Linear slope, momentum (for windows ≥4)
- **Global Context**: Comparison to historical mean, global trend slope
- **Change Metrics**: Mean and standard deviation of differences

#### 2. LightGBM Ensemble
- Multiple LightGBM models with varied hyperparameters
- Ensemble weights assigned via softmax based on validation MSE
- Parallel training for efficiency

#### 3. Linear Regression Model
- Simple linear trend model using time indices as input
- Captures overall time series direction

#### 4. Hybrid Combination
- Weighted average of ensemble and linear predictions
- Alpha parameter automatically tuned using validation performance

### Installation Requirements

```bash
pip install numpy lightgbm scikit-learn asyncio
```

### Usage Example

```python
import numpy as np
from hybrid_predictor import HybridPredictor

# Generate sample time series data
np.random.seed(42)
n_points = 200
trend = np.linspace(0, 10, n_points)
seasonality = 5 * np.sin(2 * np.pi * np.arange(n_points) / 50)
noise = np.random.normal(0, 0.5, n_points)
series = trend + seasonality + noise

# Initialize and train predictor
async def main():
    predictor = HybridPredictor(look_back=15, n_models=5)
    
    # Prepare dataset (80% train, 20% test)
    await predictor.prepare_dataset(series, train_ratio=0.8)
    
    # Train both LightGBM ensemble and Linear Regression
    await predictor.train()
    
    # Generate hybrid forecasts
    forecast_steps = 10
    hybrid_predictions = await predictor.hybrid_forecast(forecast_steps)
    
    print(f"Hybrid forecast for next {forecast_steps} steps:")
    print(hybrid_predictions)

# Run the async function
import asyncio
asyncio.run(main())
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `look_back` | 15 | Number of historical points used for feature creation |
| `n_models` | 10 | Number of LightGBM models in ensemble |
| `n_estimators` | 100 | Base number of trees for LightGBM |
| `learning_rate` | 0.05 | Learning rate for LightGBM |

### Training Process

1. **Data Preparation**: Chronological split into training and testing sets
2. **Feature Extraction**: Create features from rolling windows
3. **Parallel Training**: 
   - LightGBM ensemble models trained concurrently
   - Linear regression trained on time indices
4. **Weight Calculation**: 
   - Ensemble weights based on validation MSE
   - Hybrid alpha based on relative model performance

### Asynchronous Operations

All training and forecasting methods use async/await patterns for:
- Concurrent LightGBM model training
- Parallel prediction generation
- Efficient resource utilization
- Saving time when deployed on server. 

### Output Metrics

The hybrid model automatically balances between:
- LightGBM ensemble (captures complex patterns)
- Linear regression (captures overall trend)

### Advantages

1. **Robustness**: Multiple models reduce overfitting risk
2. **Adaptability**: Automatic weighting adapts to data characteristics
3. **Efficiency**: Asynchronous operations maximize hardware utilization
4. **Interpretability**: Linear component provides simple trend insight
5. **Accuracy**: Feature engineering captures both local and global patterns

### Limitations

- Requires sufficient historical data (minimum `look_back` points)
- Assumes stationarity in feature relationships
- Linear component may underperform on highly nonlinear series
- Asynchronous operations require proper event loop management

### Extensions

The code can be extended by:
- Adding additional feature types
- Incorporating other model types into the ensemble
- Implementing more sophisticated weight optimization
- Adding confidence intervals to predictions
- Supporting exogenous variables

# License

This implementation is provided for educational and research purposes. Modify and distribute as needed with appropriate attribution.