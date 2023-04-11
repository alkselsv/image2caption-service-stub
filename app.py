from fastapi import FastAPI, APIRouter
from fastapi import File, UploadFile
import uvicorn
from model import Model
from PIL import Image
import logging
import io

logging.basicConfig(level = logging.INFO)

app = FastAPI()
router = APIRouter()
model = Model()

@router.get("/")
async def home():
  return {"message": "machine learning service"}

@router.post("/predict")
async def data(file: UploadFile = File(...)):
  res = {}
  image_bytes = await file.read()
  img = Image.open(io.BytesIO(image_bytes))
  res['caption'] = model.predict_caption(img)
  return res

app.include_router(router)

if __name__ == "__main__":
  uvicorn.run("app:app", reload=True, port=8080, host="0.0.0.0")