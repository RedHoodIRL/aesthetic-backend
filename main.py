from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from PIL import Image
import io
import torch
import clip

app = FastAPI()

# Permitir llamadas desde tu app móvil
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Podrías restringirlo más adelante
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Descripciones "estéticas" que queremos medir
aesthetic_descriptions = [
    "cuerpo musculoso",
    "espalda en forma de V",
    "cintura delgada",
    "buena simetría",
    "físico proporcionado",
    "físico desequilibrado",
    "falta de masa muscular",
    "hombros pequeños",
]

text_tokens = clip.tokenize(aesthetic_descriptions).to(device)

@app.post("/evaluate")
async def evaluate(
    front: Annotated[UploadFile, File()],
    side: Annotated[UploadFile, File()],
    back: Annotated[UploadFile, File()]
):
    # Convertir imágenes a PIL
    images = []
    for file in [front, side, back]:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        images.append(preprocess(img).unsqueeze(0).to(device))  # Preprocesar para CLIP

    image_features = torch.cat([model.encode_image(img) for img in images])
    image_features /= image_features.norm(dim=-1, keepdim=True)

    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Similaridad entre imagen y textos
    similarities = image_features @ text_features.T  # shape: (3 imágenes, n frases)
    mean_sim = similarities.mean(dim=0)

    top_scores, top_indices = mean_sim.topk(3)
    selected_phrases = [aesthetic_descriptions[i] for i in top_indices]

    # Calcular score subjetivo entre 0–100 basado en "positivas"
    score = int((mean_sim[0] + mean_sim[1] + mean_sim[2] + mean_sim[4]) / 4 * 100)

    feedback = f"Tu físico se percibe como: {', '.join(selected_phrases)}."

    return {
        "score": score,
        "feedback": feedback
    }


    return {
        "score": score,
        "feedback": feedback
    }
