from fastapi import APIRouter, UploadFile, File
from src.services.call_service import process_audio

router = APIRouter()

@router.post("/process-audio/")
async def process_audio_endpoint(file: UploadFile = File(...)):
    return await process_audio(file) 