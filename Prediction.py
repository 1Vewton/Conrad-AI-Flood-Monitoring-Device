import lightgbm
import numpy as np
import logging
import asyncio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def generate_trendy_data(n=300, slope=0.1, noise_std=1.0):
    """
    Generate noisy linear trend data for testing.
    """
    trend = np.array([100.0]*n)
    noise = np.random.normal(0, noise_std, n)
    return trend + noise

logger = logging.getLogger("Predictor")

def create_features(window, reference_series):
    """
    Create features from a window in a time series.
    reference_series should always be only the historical/training part up to current window.
    """
    features = []
    # Window-based features
    features.extend([
        window[0],                # first value in window
        window[len(window)//2],   # middle value
        window[-1]                # last value
    ])
    features.extend([
        np.mean(window),          # mean in window
        np.std(window),           # std in window
        window[-1] - window[0]    # delta in window
    ])
    if len(window) > 1:
        diff_features = np.diff(window)
        features.extend([
            np.mean(diff_features),    # mean change in window
            np.std(diff_features)      # change std
        ])
    if len(window) >= 3:
        x = np.arange(len(window))
        features.append(np.polyfit(x, window, 1)[0])  # linear slope of window
    # "Global" features
    ref_mean = np.mean(reference_series)
    window_mean = np.mean(window)
    features.extend([
        window_mean - ref_mean,            # difference from global mean
        window_mean / (ref_mean + 1e-8)    # ratio to global mean
    ])
    global_x = np.arange(len(reference_series))
    global_slope = np.polyfit(global_x, reference_series, 1)[0]
    features.append(global_slope)
    if len(window) >= 4:
        momentum = window[-1] - 2 * window[-2] + window[-3]
        features.append(momentum)
    return features

class HybridPredictor:
    """
    Hybrid predictor combining LightGBM and Linear Regression for time series forecasting.
    """
    def __init__(self, look_back=15, n_models=10, n_estimators=100, learning_rate=0.05):
        self.look_back = look_back
        self.n_models = n_models
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.models = [None] * self.n_models
        self.models_score = [None] * self.n_models
        self.models_weight = None
        self.linreg_model = None
        self.alpha = 0.5   # Hybrid weight, will be adjusted after training

    async def prepare_dataset(self, series, train_ratio=0.8):
        """
        Prepare chronological train/test split and indices for rolling forecast.
        """
        def sync_prepare():
            X, y = [], []
            for i in range(self.look_back, len(series)):
                X.append(create_features(series[i-self.look_back:i], series[:i]))
                y.append(series[i])
            X = np.array(X)
            y = np.array(y)
            split = int(len(y) * train_ratio)
            train_X, train_y = X[:split], y[:split]
            test_X, test_y = X[split:], y[split:]
            train_series = series[:split+self.look_back]   # train sequence for rolling
            test_series = series[split+self.look_back:]    # test sequence for rolling
            train_indices = np.arange(split + self.look_back) # Indices for linear regression
            return train_X, train_y, test_X, test_y, train_series, test_series, train_indices
        loop = asyncio.get_event_loop()
        self.data_X, self.data_y, self.test_X, self.test_y, self.train_series, self.test_series, self.train_indices = await loop.run_in_executor(None, sync_prepare)

    def get_model_weights(self):
        """
        Get softmax-like weights for ensemble based on LightGBM model MSE.
        """
        mse_scores = [score["mse"] for score in self.models_score]
        min_mse = np.min(mse_scores)
        min_mse = min_mse if min_mse > 0 else 1e-8
        quality_scores = [min_mse/(score['mse']+1e-8) for score in self.models_score]
        exp_scores = np.exp(quality_scores - np.max(quality_scores))
        self.models_weight = exp_scores / np.sum(exp_scores)

    async def train(self):
        """
        Train LightGBM ensemble and Linear regression in parallel.
        """
        loop = asyncio.get_event_loop()
        # Train LightGBM ensemble
        async def train_lgbm(i):
            def train_sync(i):
                train_dataset = lightgbm.Dataset(self.data_X, label=self.data_y)
                test_dataset = lightgbm.Dataset(self.test_X, label=self.test_y)
                params = {
                    "objective": "regression",
                    "n_estimators": self.n_estimators + i * 10,
                    "metric": "l2",
                    "num_leaves": 31 + (i % 5) * 5,
                    "learning_rate": self.lr + 0.001 * i,
                    "random_state": 42 + i,
                    "verbose": -1
                }
                model = lightgbm.train(params=params, train_set=train_dataset, valid_sets=[test_dataset])
                self.models[i] = model
                test_result = model.predict(self.test_X)
                mse = mean_squared_error(self.test_y, test_result)
                mae = mean_absolute_error(self.test_y, test_result)
                score = {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse), 'model_idx': i}
                self.models_score[i] = score
            await loop.run_in_executor(None, train_sync, i)

        # Train all LightGBM models concurrently
        await asyncio.gather(*(train_lgbm(i) for i in range(self.n_models)))
        self.get_model_weights()

        # Train Linear Regression on train series index (one-dimensional time input)
        def train_linreg():
            X_lr = self.train_indices.reshape(-1, 1)
            y_lr = self.train_series
            return LinearRegression().fit(X_lr, y_lr)
        self.linreg_model = await loop.run_in_executor(None, train_linreg)

        # Adjust hybrid alpha automatically based on validation error
        # Use validation/test MSE: alpha is set so that "best" performing model gets highest weight
        # Smaller error => bigger alpha weight (normalized to [0,1])
        lgbm_ensemble_mse = np.mean([score['mse'] for score in self.models_score])
        linreg_val_pred = self.linreg_model.predict(
            np.arange(len(self.train_series), len(self.train_series) + len(self.test_y)).reshape(-1, 1)
        )
        linreg_mse = mean_squared_error(self.test_y, linreg_val_pred)
        # Avoid zero division
        total = lgbm_ensemble_mse + linreg_mse + 1e-8
        self.alpha = (linreg_mse / total) if (linreg_mse + lgbm_ensemble_mse) > 0 else 0.5
        # This means alpha is higher if LightGBM error is lower; hybrid prediction gives more weight to better predictor

    async def ensemble_forecast(self, steps):
        """
        Asynchronous LightGBM rolling forecast (multi-step ahead, beyond train).
        """
        predictions = np.zeros(steps)
        loop = asyncio.get_event_loop()

        async def single_model_forecast(idx):
            model = self.models[idx]
            weight = self.models_weight[idx]
            rolling_win = self.train_series[-self.look_back:].copy()
            pred = []
            for step in range(steps):
                features = create_features(rolling_win, self.train_series)
                next_val = await loop.run_in_executor(None, model.predict, np.array(features).reshape(1, -1))
                next_val = next_val[0]
                pred.append(next_val * weight)
                rolling_win = np.append(rolling_win[1:], next_val)
            return np.array(pred)

        tasks = [single_model_forecast(i) for i in range(self.n_models)]
        results = await asyncio.gather(*tasks)
        for res in results:
            predictions += res
        return predictions

    async def linear_forecast(self, steps):
        """
        Asynchronous Linear Regression forecast for future time indices.
        """
        start_idx = len(self.train_series)
        indices = np.arange(start_idx, start_idx + steps).reshape(-1,1)
        loop = asyncio.get_event_loop()
        # Since scikit-learn predict is fast, can await directly
        pred = await loop.run_in_executor(None, self.linreg_model.predict, indices)
        return pred

    async def hybrid_forecast(self, steps):
        """
        Hybrid prediction: weighted sum of async LightGBM and Linear Regression predictions.
        """
        lgbm_pred, linreg_pred = await asyncio.gather(
            self.ensemble_forecast(steps),
            self.linear_forecast(steps)
        )
        # Hybrid alpha chosen after training by validation error for automatic model weighting
        return self.alpha * lgbm_pred + (1.0 - self.alpha) * linreg_pred

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    test_data = generate_trendy_data()
    pred = HybridPredictor()
    # Prepare dataset (chronological split)
    loop.run_until_complete(pred.prepare_dataset(test_data, train_ratio=0.8))
    loop.run_until_complete(pred.train())
    forecast_steps = min(120, len(pred.test_series))

    # Asynchronous prediction
    lgbm_pred, linreg_pred, hybrid_pred = loop.run_until_complete(asyncio.gather(
        pred.ensemble_forecast(forecast_steps),
        pred.linear_forecast(forecast_steps),
        pred.hybrid_forecast(forecast_steps)
    ))

    # Plot all
    plt.plot(np.arange(len(test_data)), test_data, label='All Data', alpha=0.5)
    plt.plot(
        np.arange(len(test_data)-forecast_steps, len(test_data)),
        test_data[-forecast_steps:], 'b', label='True Future'
    )
    plt.plot(
        np.arange(len(test_data)-forecast_steps, len(test_data)),
        lgbm_pred, 'orange', label='LightGBM Forecast'
    )
    plt.plot(
        np.arange(len(test_data)-forecast_steps, len(test_data)),
        linreg_pred, 'g', label='Linear Regression Forecast'
    )
    plt.plot(
        np.arange(len(test_data)-forecast_steps, len(test_data)),
        hybrid_pred, 'r', label='Hybrid Forecast (α={:.2f})'.format(pred.alpha)
    )
    plt.legend()
    plt.show()

    # Print MSE and MAE for each method
    print("Test size:", forecast_steps)
    print("LightGBM MSE:", mean_squared_error(test_data[-forecast_steps:], lgbm_pred))
    print("LinearReg MSE:", mean_squared_error(test_data[-forecast_steps:], linreg_pred))
    print("Hybrid MSE:", mean_squared_error(test_data[-forecast_steps:], hybrid_pred))
    print("LightGBM MAE:", mean_absolute_error(test_data[-forecast_steps:], lgbm_pred))
    print("LinearReg MAE:", mean_absolute_error(test_data[-forecast_steps:], linreg_pred))
    print("Hybrid MAE:", mean_absolute_error(test_data[-forecast_steps:], hybrid_pred))
    print("Hybrid α (weighted by validation error): {:.2f}".format(pred.alpha))