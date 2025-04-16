from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from sarima_model import train_and_predict as sarima_forecast
from prophet_model import process_category as prophet_forecast
from gradient_boosting_model import process_category as gb_forecast

app = FastAPI()

@app.post("/predict/")
async def predict(model: str, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file, parse_dates=["date"])
        if model == "sarima":
            results = sarima_forecast(df)
        elif model == "prophet":
            results = {
                cat: prophet_forecast(df, cat)
                for cat in df["category"].unique()
            }
        elif model == "gb":
            results = {
                cat: gb_forecast(df[df["category"] == cat])
                for cat in df["category"].unique()
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))