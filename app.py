from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Must match the exact order used during training
CLASS_NAMES = ['Adenocarcinoma', 'Benign', 'Squamous Cell Carcinoma']
WEIGHTS_PATH = "lung_cancer_detection_using_DenseNet.weights.h5"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
IMG_SIZE = (224, 224)

# ── Model (loaded once at startup) ───────────────────────────────────────────
model: tf.keras.Model = None


def build_model() -> tf.keras.Model:
    """Reconstruct the DenseNet121-based classifier and load saved weights.
    Architecture must mirror the notebook exactly (Sequential + trimmed base sub-model).
    """
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    # Trim the base at pool2_pool — exactly as done in the notebook
    trimmed_base = tf.keras.Model(
        inputs=base.input,
        outputs=base.get_layer("pool2_pool").output,
    )

    # Wrap in Sequential so the layer graph matches the saved weights
    m = Sequential([
        trimmed_base,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(CLASS_NAMES), activation="softmax"),
    ])

    # No skip_mismatch — architectures now match, so every layer must load cleanly
    m.load_weights(WEIGHTS_PATH)
    m.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Model loaded successfully.")
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts; release on shutdown."""
    global model
    model = build_model()
    yield
    del model
    logger.info("Model released.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lung Cancer Detection API",
    description=(
        "Detect lung cancer from histopathological images "
        "using a pre-trained DenseNet121 model."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def validate_image_file(file: UploadFile) -> None:
    """Raise HTTP 415 if the uploaded file is not a supported image type."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes → preprocessed numpy array ready for DenseNet."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")  
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return img_array


def run_inference(img_array: np.ndarray) -> dict:
    """Run model prediction and return the top class with its confidence."""
    predictions = model.predict(img_array, verbose=0)  # shape: (1, num_classes)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index]) * 100
    
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": f"{confidence:.2f}%",
        
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "message": (
            "Lung Cancer Detection API is running. "
            "POST an image to /predict (single) or /predict_batch (multiple)."
        )
    }





@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Single histopathological image (JPEG/PNG).")
):
    """
    Classify a single lung tissue image as one of:
    - adenocarcinoma
    - benign
    - squamous cell carcinoma
    """
    validate_image_file(file)
    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)
        return run_inference(img_array)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/predict_batch", tags=["Inference"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="Multiple histopathological images.")
):
    """
    Classify multiple lung tissue images in one request.
    Returns a list of predictions in the same order as the uploaded files.
    """
    results = []
    for file in files:
        validate_image_file(file)
        try:
            image_bytes = await file.read()
            img_array = preprocess_image(image_bytes)
            result = run_inference(img_array)
            results.append({"filename": file.filename, **result})
        except HTTPException as e:
            results.append({"filename": file.filename, "error": e.detail})
        except Exception as e:
            logger.exception(f"Batch prediction failed for {file.filename}.")
            results.append({"filename": file.filename, "error": str(e)})
    return {"predictions": results}