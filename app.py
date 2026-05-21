from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import easyocr
import re
import numpy as np
import os

app = FastAPI()

# Variables globales
model = None
processor = None
reader = None


@app.get("/")
def root():
    return {"status": "ok"}


@app.on_event("startup")
def load_models():
    global model, processor, reader

    token = os.getenv("HF_TOKEN")

    print("⏳ Cargando modelo CLIP...")

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        token=token
    )

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        token=token
    )

    print("✅ Modelo CLIP cargado")

    print("⏳ Cargando OCR...")
    reader = easyocr.Reader(['es', 'en'])
    print("✅ OCR cargado")

def clasificar_binario(imagen: Image.Image, clase_positiva: str, clase_negativa: str):
    inputs = processor(
        text=[clase_positiva, clase_negativa],
        images=imagen,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    return round(float(probs[0]) * 100, 2)


def clasificar_subclases(imagen: Image.Image, subclases: list[str]):
    inputs = processor(
        text=subclases,
        images=imagen,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    resultados = [
        {"clase": clase, "confianza": round(float(prob) * 100, 2)}
        for clase, prob in zip(subclases, probs)
    ]
    return sorted(resultados, key=lambda x: x['confianza'], reverse=True)


@app.post("/clasificar")
async def clasificar(imagen: UploadFile = File(...)):

    contenido = await imagen.read()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")

    # ============================================================
    # 1. CLASIFICACIÓN PRINCIPAL
    # ============================================================
    confianza_libro = clasificar_binario(
        img,
        "a book with a cover and pages",
        "this is not a book"
    )

    confianza_paquete = clasificar_binario(
        img,
        "a cardboard box or shipping package",
        "this is not a package or box"
    )

    # ============================================================
    # 2. SUBCLASES
    # ============================================================
    subclases_libro = [
        "una novela con portada ilustrada",
        "un libro de tapa blanda",
        "un libro de tapa dura",
        "un libro con lomo visible",
        "un libro parado sobre una mesa",
        "un libro acostado sobre una superficie",
        "un libro con portada colorida",
        "un libro de ficción con ilustración",
        "un libro escolar o de texto",
        "un libro de cuentos o novela",
        "un libro con título en la portada",
        "un libro de literatura clásica",
        "un libro infantil con dibujos",
        "un libro universitario o académico",
        "un diccionario o enciclopedia",
        "un libro con portada de persona",
        "un libro con portada oscura",
        "un libro apilado con otros libros",
        "un libro en una estantería",
        "un comic o manga",
    ]

    subclases_paquete = [
        "una caja de cartón marrón",
        "un paquete de envío sellado con cinta",
        "una caja rectangular de cartón",
        "un paquete de delivery o courier",
        "una caja de embalaje corrugado",
        "un paquete envuelto en papel marrón",
        "una caja de cartón cerrada",
        "un paquete con etiqueta de envío",
        "una caja de cartón abierta",
        "un sobre de burbujas amarillo",
        "una caja blanca de cartón",
        "un paquete de plástico sellado",
        "una caja con cinta adhesiva marrón",
        "un paquete con logos de courier",
        "una caja de mudanza grande",
    ]

    detalle_libro   = clasificar_subclases(img, subclases_libro)
    detalle_paquete = clasificar_subclases(img, subclases_paquete)

    # ============================================================
    # 3. CATEGORÍA FINAL
    # ============================================================
    umbral = 60
    if confianza_libro >= umbral and confianza_libro > confianza_paquete:
        categoria = "LIBRO"
        confianza_final = confianza_libro
    elif confianza_paquete >= umbral and confianza_paquete > confianza_libro:
        categoria = "PAQUETE"
        confianza_final = confianza_paquete
    else:
        categoria = "DESCONOCIDO"
        confianza_final = max(confianza_libro, confianza_paquete)

    # ============================================================
    # 4. OCR - LEER TEXTO DE LA IMAGEN
    # ============================================================
    img_array = np.array(img)
    resultados_ocr = reader.readtext(img_array)

    textos_detectados = []
    textos_raw = []

    for (bbox, texto, confianza_ocr) in resultados_ocr:
        textos_detectados.append({
            "texto": texto,
            "confianza": round(confianza_ocr * 100, 2)
        })
        textos_raw.append(texto)

    # Filtrar textos con alta confianza (probables títulos)
    textos_relevantes = [
        t for t in textos_detectados
        if t["confianza"] >= 70  # solo textos con buena confianza
    ]

    # El título suele ser el texto más largo y con más confianza
    titulo_detectado = None
    if textos_relevantes:
        # ordenar por longitud del texto para encontrar el título
        candidatos_titulo = sorted(
            textos_relevantes,
            key=lambda x: len(x["texto"]),
            reverse=True
        )
        titulo_detectado = candidatos_titulo[0]["texto"]

    # ============================================================
    # 5. RESPUESTA JSON
    # ============================================================
    return {
        "categoria": categoria,
        "confianza": confianza_final,
        "clasificacion_principal": {
            "libro": confianza_libro,
            "paquete": confianza_paquete
        },
        "subclases_libro": detalle_libro[:5],
        "subclases_paquete": detalle_paquete[:5],
        "ocr": {
            "titulo_detectado": titulo_detectado,
            "todos_los_textos": textos_detectados
        }
    }


def extraer_tracking(imagen: Image.Image):
    img_array = np.array(imagen)
    
    # Extraer todo el texto - resultados son tuplas (bbox, texto, confianza)
    resultados = reader.readtext(img_array)
    
    textos = []
    textos_raw = []  # solo los strings de texto
    
    for (bbox, texto, confianza) in resultados:
        textos.append({
            "texto": texto,
            "confianza": round(confianza * 100, 2)
        })
        textos_raw.append(texto)  # ← guardamos solo el string
    
    # Buscar código de tracking con regex
    patrones_tracking = [
        r'\b[A-Z]{2,4}\d{8,12}\b',    # WYB445884472
        r'\b[A-Z]{1,3}\d{9,15}\b',    # PE123456789
        r'\b\d{10,15}\b',              # solo números largos
        r'\b[A-Z0-9]{10,20}\b',        # alfanumérico largo
    ]
    
    trackings_encontrados = []
    texto_completo = " ".join(textos_raw)  # ← usamos la lista de strings
    
    for patron in patrones_tracking:
        matches = re.findall(patron, texto_completo)
        for match in matches:
            if match not in trackings_encontrados:
                trackings_encontrados.append(match)
    
    return {
        "trackings": trackings_encontrados,
        "tracking_principal": trackings_encontrados[0] if trackings_encontrados else None,
        "todo_el_texto": textos
    }


@app.post("/extraer-tracking")
async def extraer_tracking_endpoint(imagen: UploadFile = File(...)):
    contenido = await imagen.read()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")
    
    resultado = extraer_tracking(img)
    
    return {
        "tracking_principal": resultado["tracking_principal"],
        "todos_los_trackings": resultado["trackings"],
        "texto_detectado": resultado["todo_el_texto"]
    }