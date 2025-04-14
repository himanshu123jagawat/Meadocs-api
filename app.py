import os
import wave
import json
import re
import numpy as np
import torch
import faiss
import cv2
import pandas as pd
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import clip
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from fuzzywuzzy import fuzz
import pdfplumber
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from sentence_transformers import SentenceTransformer

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Photo Module Setup (CLIP for images)
photo_clip_model, photo_preprocess = clip.load("ViT-B/32", device=device)
photo_index = None
photo_paths = []

# Video Module Setup (CLIP for video frames)
video_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
video_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
video_index = faiss.IndexFlatL2(512)
video_map = {}
video_frame_store = {}
video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
])

# Audio Module Setup (Vosk)
UPLOAD_FOLDER = "uploads"
TRANSCRIPTIONS_FOLDER = os.path.join(UPLOAD_FOLDER, "transcriptions")
MODEL_PATH = "/Users/lakshyadivekar/Downloads/vosk-model-en-us-0.42-gigaspeech"
SUPPORTED_FORMATS = (".mp3", ".wav", ".m4a", ".flac")
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

# Document Module Setup (SentenceTransformer)
doc_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = []
doc_filenames = []
doc_index = faiss.IndexFlatL2(384)

# Pydantic models for request validation
class ProcessRequest(BaseModel):
    media_type: str  # "photo", "video", "audio", "document"
    folder: str      # Directory path

class SearchRequest(BaseModel):
    media_type: str
    prompt: str
    top_k: Optional[int] = 5

# Photo Module Functions
def get_image_paths(directory):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return [os.path.join(root, file) for root, _, files in os.walk(directory)
            for file in files if os.path.splitext(file)[1].lower() in image_extensions]

def preprocess_images(image_paths):
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            image = photo_preprocess(Image.open(path)).unsqueeze(0).to(device)
            images.append(image)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return images, valid_paths

def encode_images(images):
    with torch.no_grad():
        features = torch.cat([photo_clip_model.encode_image(img) for img in images])
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def build_photo_index(image_features):
    d = image_features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(image_features)
    return index

def search_photos(query, top_k=5):
    query_tokenized = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = photo_clip_model.encode_text(query_tokenized)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()
    distances, indices = photo_index.search(text_features, top_k)
    return [{"path": photo_paths[i], "score": float(distances[0][j]), "mime": "image/jpeg"} for j, i in enumerate(indices[0])]

# Video Module Functions
def extract_frames(video_path, output_folder, frame_interval=3):
    video_name = os.path.basename(video_path)
    video_frame_folder = os.path.join(output_folder, video_name)
    if os.path.exists(video_frame_folder) and os.listdir(video_frame_folder):
        return [(os.path.join(video_frame_folder, f), float(f.split('_')[1].split('.')[0]))
                for f in os.listdir(video_frame_folder) if f.startswith('frame_')]
    os.makedirs(video_frame_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    extracted_frames = []
    success, image = cap.read()
    while success:
        timestamp = frame_count / fps
        if int(timestamp) % frame_interval == 0:
            frame_filename = os.path.join(video_frame_folder, f"frame_{int(timestamp)}.jpg")
            cv2.imwrite(frame_filename, image)
            extracted_frames.append((frame_filename, timestamp))
        success, image = cap.read()
        frame_count += 1
    cap.release()
    return extracted_frames

def encode_frame(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = video_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = video_clip_model.get_image_features(image_tensor)
    features /= torch.norm(features, dim=-1, keepdim=True)
    return features.cpu().numpy()

def search_video(query, top_k=3):
    inputs = video_clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = video_clip_model.get_text_features(**inputs)
    text_features /= torch.norm(text_features, dim=-1, keepdim=True)
    query_embedding = text_features.cpu().numpy()
    D, I = video_index.search(query_embedding, top_k)
    if I[0][0] == -1:
        return []
    results = []
    for idx in I[0]:
        if idx in video_map:
            video_path, timestamp = video_map[idx]
            results.append({"video_path": video_path, "timestamp": timestamp, "score": float(D[0][idx]), "mime": "video/mp4"})
    return results

# Audio Module Functions
def sanitize_filename(filename):
    return re.sub(r'[^\w\-_. ]', '', filename)

def convert_to_wav(audio_path):
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    if not os.path.exists(wav_path):
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(audio_path):
    model = Model(MODEL_PATH)
    if not audio_path.endswith(".wav"):
        audio_path = convert_to_wav(audio_path)
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    transcript = ""
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript += result.get("text", "") + " "
    final_result = json.loads(rec.FinalResult())
    transcript += final_result.get("text", "")
    return transcript.strip()

def save_transcription(filename, text):
    base_name = sanitize_filename(os.path.splitext(filename)[0])
    path = os.path.join(TRANSCRIPTIONS_FOLDER, f"{base_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def find_original_audio_file(base_name):
    for ext in SUPPORTED_FORMATS:
        path = os.path.join(UPLOAD_FOLDER, base_name + ext)
        if os.path.exists(path):
            return path
    return None

def search_audio(query, top_k=5):
    matches = []
    for text_file in os.listdir(TRANSCRIPTIONS_FOLDER):
        if text_file.endswith(".txt"):
            with open(os.path.join(TRANSCRIPTIONS_FOLDER, text_file), "r", encoding="utf-8") as f:
                transcript = f.read()
            score = fuzz.partial_ratio(query.lower(), transcript.lower())
            audio_file = text_file.replace(".txt", "")
            matches.append((audio_file, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    results = []
    for file, score in matches[:top_k]:
        audio_path = find_original_audio_file(file)
        results.append({
            "file": file,
            "score": score,
            "path": audio_path,
            "mime": "audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav"
        })
    return results

# Document Module Functions
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def extract_text_from_excel(file_path):
    df = pd.read_excel(file_path, sheet_name=None)
    text = ""
    for sheet_name, sheet_data in df.items():
        text += f"\nSheet: {sheet_name}\n"
        text += sheet_data.to_string(index=False)
    return text

def extract_images_from_pdf(file_path):
    doc = fitz.open(file_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
    return images

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def process_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    try:
        if ext == ".pdf":
            extracted_text += extract_text_from_pdf(file_path)
            images = extract_images_from_pdf(file_path)
            for img in images:
                extracted_text += extract_text_from_image(img) + "\n"
        elif ext == ".docx":
            extracted_text += extract_text_from_docx(file_path)
        elif ext == ".txt":
            extracted_text += extract_text_from_txt(file_path)
        elif ext == ".xlsx":
            extracted_text += extract_text_from_excel(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return extracted_text.strip()

def search_documents(query, top_k=5):
    query_vector = doc_model.encode([query])[0]
    query_vector = np.array([query_vector], dtype=np.float32)
    distances, indices = doc_index.search(query_vector, top_k)
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        results.append({
            "filename": doc_filenames[idx],
            "score": float(distances[0][i]),
            "mime": "application/pdf" if doc_filenames[idx].endswith(".pdf") else "text/plain"
        })
    return results

# FastAPI Endpoints
@app.post("/process")
async def process_media(req: ProcessRequest):
    global photo_index, photo_paths, video_index, video_map, video_frame_store, doc_texts, doc_filenames, doc_index
    media_type = req.media_type.lower()
    folder = req.folder

    if not os.path.exists(folder):
        raise HTTPException(status_code=400, detail=f"Folder '{folder}' does not exist")

    if media_type == "photo":
        image_paths = get_image_paths(folder)
        if not image_paths:
            raise HTTPException(status_code=400, detail="No images found in the folder")
        images, photo_paths = preprocess_images(image_paths)
        image_features = encode_images(images)
        photo_index = build_photo_index(image_features)
        return {"status": "success", "message": f"Indexed {len(photo_paths)} images"}

    elif media_type == "video":
        video_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            raise HTTPException(status_code=400, detail="No videos found in the folder")
        frame_folder = os.path.join(folder, "extracted_frames")
        all_embeddings = []
        for video_path in video_files:
            frames = extract_frames(video_path, frame_folder)
            for frame, timestamp in frames:
                embedding = encode_frame(frame)
                all_embeddings.append(embedding)
                video_map[len(video_map)] = (video_path, timestamp)
                video_frame_store[len(video_frame_store)] = frame
        if all_embeddings:
            video_index.add(np.vstack(all_embeddings))
        return {"status": "success", "message": f"Indexed {len(video_files)} videos"}

    elif media_type == "audio":
        audio_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(SUPPORTED_FORMATS)]
        if not audio_files:
            raise HTTPException(status_code=400, detail="No audio files found in the folder")
        for audio_path in audio_files:
            filename = os.path.basename(audio_path)
            transcript = transcribe_audio(audio_path)
            save_transcription(filename, transcript)
        return {"status": "success", "message": f"Processed {len(audio_files)} audio files"}

    elif media_type == "document":
        doc_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".pdf", ".docx", ".txt", ".xlsx"))]
        if not doc_files:
            raise HTTPException(status_code=400, detail="No documents found in the folder")
        for file_path in doc_files:
            text = process_document(file_path)
            if text:
                doc_texts.append(text)
                doc_filenames.append(file_path)
                vector = doc_model.encode([text])[0]
                doc_index.add(np.array([vector], dtype=np.float32))
        return {"status": "success", "message": f"Indexed {len(doc_files)} documents"}

    raise HTTPException(status_code=400, detail="Invalid media type")

@app.post("/search")
async def search_media(req: SearchRequest):
    media_type = req.media_type.lower()
    prompt = req.prompt
    top_k = req.top_k

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if media_type == "photo":
        if photo_index is None:
            raise HTTPException(status_code=400, detail="Photos not indexed. Process a folder first.")
        results = search_photos(prompt, top_k)
        return {"status": "success", "results": results}

    elif media_type == "video":
        if not video_map:
            raise HTTPException(status_code=400, detail="Videos not indexed. Process a folder first.")
        results = search_video(prompt, top_k)
        return {"status": "success", "results": results}

    elif media_type == "audio":
        results = search_audio(prompt, top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No matching audio found")
        return {"status": "success", "results": results}

    elif media_type == "document":
        if not doc_filenames:
            raise HTTPException(status_code=400, detail="Documents not indexed. Process a folder first.")
        results = search_documents(prompt, top_k)
        return {"status": "success", "results": results}

    raise HTTPException(status_code=400, detail="Invalid media type")

@app.get("/serve")
async def serve_file(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    from fastapi.responses import FileResponse
    return FileResponse(path)

@app.get("/health")
async def health():
    return {"status": "Backend is running"}

if __name__ == "__main__":   
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)