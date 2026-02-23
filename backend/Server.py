from fastapi import FastAPI, HTTPException
import logger_config
from DataModel import WaterLevelDataTest
from Prediction import HybridPredictor
from typing import List
import logging
import uvicorn

logger = logging.getLogger("Server")
# Tags
Tags = [
    {"name": "Prediction Test",
     "description": "API to test the prediction function"}
]
# Application starting
app = FastAPI(
    title="SmartDrain Backend",
    openapi_tags=Tags
)
# APIs
# Predictor test interface
@app.post("/test/predict")
async def test_predict(data: WaterLevelDataTest) -> List[float]:
    try:
        # Predictor
        predictor = HybridPredictor()
        await predictor.prepare_dataset(data.water_level)
        logger.info("Dataset Preparation Successful")
        await predictor.train()
        logger.info("Training Successful")
        result = await predictor.hybrid_forecast(data.forecast_steps)
        logger.info("Forecasting Successful")
        # Turn it into List[float]
        return result.tolist()
    except Exception as e:
        # Error handling
        logger.error(f"An error occurred due to {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred due to {e}")

if __name__ == "__main__":
    logger_config.setup_logging()
    # Start up service
    logger.info("Starting Server on port 8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)

