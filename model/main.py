import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms, utils

import numpy as np
from PIL import Image
import os
import math
import torchvision.models as models
from torchvision.ops import generalized_box_iou
from models_lite import FullModel
from utils import box_cxcywh_to_xyxy, box_iou, ciou_loss_xyxy, get_objects_on_image

# FastAPI imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi import UploadFile, File
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
embedding_size = 256
hidden_embedding_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 4
classes_count = 1

'''
embedding_size = 192
hidden_embedding_size = 384
num_heads = 6
num_encoder_layers = 4
num_decoder_layers = 3
classes_count = 1

model_path = "detr_face_detection_model_lite.pth"

model = FullModel(embedding_size, hidden_embedding_size, num_heads, num_encoder_layers, num_decoder_layers, classes_count, 100, 0.0)
model.to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Model loaded from {model_path}")
model.eval()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "This is a root of model server. Use /predict endpoint to get predictions."}

@app.post("/predict")
async def predict(tensor_file: UploadFile = File(...)):
    tensor_bytes = await tensor_file.read()
    image = torch.load(io.BytesIO(tensor_bytes), map_location=device)
    with torch.no_grad():
        boxes, classes = get_objects_on_image(image, model, device)
    return {"boxes": boxes, "classes": classes.tolist()}

@app.get("/health")
def health():
    return {"status": "ok"}