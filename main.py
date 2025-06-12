from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import random

app = FastAPI()

# Permitir peticiones desde la app móvil
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir esto a tu IP si lo prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluate")
async def evaluate(
    front: Annotated[UploadFile, File()],
    side: Annotated[UploadFile, File()],
    back: Annotated[UploadFile, File()],
):
    # Simular procesamiento de IA con un score aleatorio
    score = random.randint(60, 95)
    feedbacks = [
        "Buena proporción, mejora la definición abdominal.",
        "Trabaja en la simetría de la espalda.",
        "Buen desarrollo de torso, mejora hombros.",
        "Piernas balanceadas, foco en glúteos.",
    ]
    feedback = random.choice(feedbacks)

    return {
        "score": score,
        "feedback": feedback
    }
