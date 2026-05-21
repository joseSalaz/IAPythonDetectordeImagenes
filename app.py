from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import numpy as np
import os
import re

app = FastAPI()

# =========================
# GLOBALS (UNA SOLA VEZ)
# =========================
model = None
processor = None
reader = None


# =========================
# OPTIMIZACIÓN CPU
# =========================
torch.set_num_threads(1)


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def load_models():
    global model, processor

    token = os.getenv("HF_TOKEN")

    print("⏳ Cargando CLIP...")

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        token=token
    )
    model.eval()

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        token=token
    )

    print("✅ CLIP listo")


# =========================
# OCR LAZY LOAD (IMPORTANTE)
# =========================
def get_reader():
    global reader
    if reader is None:
        print("⏳ Cargando OCR...")
        import easyocr
        reader = easyocr.Reader(['es', 'en'])
        print("✅ OCR listo")
    return reader


# =========================
# CLIP SIMPLE (OPTIMIZADO)
# =========================
def clasificar(imagen, textos):
    inputs = processor(
        text=textos,
        images=imagen,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]
    return probs.tolist()


# =========================
# ENDPOINT PRINCIPAL
# =========================
@app.post("/clasificar")
async def clasificar_imagen(imagen: UploadFile = File(...)):

    contenido = await imagen.read()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")

    # =========================
    # 1. CLIP (REDUCIDO)
    # =========================
    clases = ["book", "package", "unknown"]

    probs = clasificar(img, clases)

    resultado = dict(zip(clases, probs))

    categoria = max(resultado, key=resultado.get)

    confianza = round(resultado[categoria] * 100, 2)

    # =========================
    # 2. SUBCLASES (REDUCIDAS)
    # =========================
    subclases = [
        "book cover",
        "hardcover book",
        "paperback book",
        "book on table",
        "cardboard box",
        "shipping package",
        "sealed box",
        "delivery package"
    ]

    sub_probs = clasificar(img, subclases)

    sub_resultados = [
        {"clase": c, "confianza": round(p * 100, 2)}
        for c, p in zip(subclases, sub_probs)
    ]

    sub_resultados = sorted(sub_resultados, key=lambda x: x["confianza"], reverse=True)

    # =========================
    # 3. OCR (SOLO SI ES NECESARIO)
    # =========================
    reader = get_reader()

    img_array = np.array(img)
    ocr_results = reader.readtext(img_array)

    textos = [
        {"texto": t, "confianza": round(c * 100, 2)}
        for (_, t, c) in ocr_results
    ]

    texto_completo = " ".join([t["texto"] for t in textos])

    # Tracking simple
    trackings = re.findall(r'\b[A-Z0-9]{10,20}\b', texto_completo)

    # =========================
    # 4. RESPUESTA FINAL
    # =========================
    return {
        "categoria": categoria,
        "confianza": confianza,
        "probabilidades": resultado,
        "subclases_top": sub_resultados[:3],
        "ocr": {
            "textos": textos,
            "tracking": trackings[0] if trackings else None
        }
    }


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"status": "ok"}