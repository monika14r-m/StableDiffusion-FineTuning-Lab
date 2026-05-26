from fastapi import APIRouter, UploadFile, File
import shutil
from src.core.pipeline import generate_image

router = APIRouter()

@router.post("/generate")
async def generate(file: UploadFile = File(...)):
    path = f"temp/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output = generate_image(path)

    return {"result": output}
