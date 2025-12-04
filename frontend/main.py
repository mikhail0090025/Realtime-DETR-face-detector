from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import text
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import requests
import io
from PIL import Image
import torch
import numpy as np

app = FastAPI()

app.mount("/templates", StaticFiles(directory="templates"), name="templates")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def root():
    return {"message": "Frontend service is up!"}

@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((240, 240), Image.BILINEAR)
    image_np = np.array(image)
    image = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    image = image / 255.0
    tensor_bytes = io.BytesIO()
    torch.save(image, tensor_bytes)
    tensor_bytes.seek(0)
    response = requests.post(
        "http://model_service:8001/predict",
        files={"tensor_file": tensor_bytes}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Model service error")
    print(response.json())
    return JSONResponse(content=response.json())

@app.get("/health")
def health():
    return {"status": "ok"}