from PIL import Image
import io
import torch
import hydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
import os
from pathlib import Path
import threading

# Constants
SESSIONS_FOLDER = "sessions"
MAIN_IMAGE_FOLDER = "images"
EMBEDDINGS_DIR = os.getenv("SAM2_EMBEDDINGS_DIR", "embeddings")

_precompute_lock = threading.Lock()

# Build model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAM2_VERSION = os.getenv("SAM2_VERSION", "2.1").strip()
SAM2_MODEL_ID = os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-large")
PRECISION_BF16 = os.getenv("SAM2_BF16", "1") in ("1", "true", "True")
torch_dtype = torch.bfloat16 if (PRECISION_BF16 and DEVICE.type == "cuda") else None
LOCAL_CKPT = os.getenv("SAM2_LOCAL_CHECKPOINT", "sam2.1_hiera_large.pt")
LOCAL_CONFIG = os.getenv("SAM2_LOCAL_CONFIG", "sam2.1_hiera_l.yaml")

# Ensure hydra state is clean and initialize config module
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module("sam2_configs", version_base="1.2")

print(f"Device: {DEVICE} - Using SAM2 model id {SAM2_MODEL_ID}")

# Build model and predictor
model = build_sam2(LOCAL_CONFIG, LOCAL_CKPT, device=DEVICE, apply_postprocessing=False)
predictor = SAM2ImagePredictor(model)

# ensure embeddings dir exists
os.makedirs(SESSIONS_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


# Utility helpers


def apply_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask_bin = (mask > 0).astype(np.uint8) * 255
    else:
        mask_bin = (mask > 0).astype(np.uint8) * 255
    h, w = mask_bin.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask_bin > 0] = (255, 255, 255)
    return out


def run_sam2_predict(
    *, image_rgb: np.ndarray, points=None, labels=None, boxes=None, multimask=True
) -> np.ndarray:
    import torch as _torch

    predictor.set_image(image_rgb)
    kwargs = {}
    if points is not None:
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 2:
            pts = pts[None, :, :]
        kwargs["point_coords"] = _torch.tensor(pts, device=DEVICE)
    if labels is not None:
        lbls = np.array(labels, dtype=np.int64)
        if lbls.ndim == 1:
            lbls = lbls[None, :]
        kwargs["point_labels"] = _torch.tensor(lbls, device=DEVICE)
    if boxes is not None:
        b = np.array(boxes, dtype=np.float32)
        if b.shape == (2, 2):
            b = np.array([b[0, 0], b[0, 1], b[1, 0], b[1, 1]], dtype=np.float32)
        if b.ndim == 1:
            b = b[None, :]
        kwargs["box"] = _torch.tensor(b, device=DEVICE)
    kwargs["multimask_output"] = multimask
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(torch_dtype == torch.bfloat16)
    ):
        masks, scores, _ = predictor.predict(**kwargs)
    # Normalize types
    if not torch.is_tensor(scores):
        scores_t = torch.tensor(scores)
    else:
        scores_t = scores
    best_idx = int(torch.argmax(scores_t).item())
    if torch.is_tensor(masks):
        best = masks[best_idx].detach().cpu().numpy()
    else:
        best = np.array(masks)[best_idx]
    mask_uint8 = (best > 0).astype(np.uint8) * 255
    return mask_uint8


# Embedding cache utilities


def _embedding_group_dir(group: str) -> str:
    return os.path.join(EMBEDDINGS_DIR, group)


def _embedding_file_path(group: str, image_name: str) -> str:
    return os.path.join(_embedding_group_dir(group), f"{image_name}.pt")


def embedding_exists(group: str, image_name: str) -> bool:
    return os.path.exists(_embedding_file_path(group, image_name))


def group_precomputed(group: str) -> bool:
    gdir = _embedding_group_dir(group)
    if not os.path.isdir(gdir):
        return False
    if os.path.exists(os.path.join(gdir, ".done")):
        return True
    return any(f.endswith(".pt") for f in os.listdir(gdir))


def save_group_done(group: str):
    with open(os.path.join(_embedding_group_dir(group), ".done"), "w") as f:
        f.write("ok")


def extract_image_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def compute_and_save_embedding(group: str, image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return False, "failed to read image"
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    emb = None
    used_attr = None
    for attr in [
        "_features",
        "features",
        "_image_embedding",
        "image_embedding",
        "_image_features",
    ]:
        if hasattr(predictor, attr):
            candidate = getattr(predictor, attr)
            if candidate is not None:
                emb = candidate
                used_attr = attr
                break
    if emb is None:
        return False, "embedding attribute not found on predictor"

    def _to_cpu(obj):
        import torch as _torch

        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [_to_cpu(v) for v in obj]
            return type(obj)(t) if not isinstance(obj, list) else t
        return obj

    emb_cpu = _to_cpu(emb)
    original_size = getattr(predictor, "original_size", (rgb.shape[0], rgb.shape[1]))
    input_size = getattr(predictor, "input_size", (rgb.shape[0], rgb.shape[1]))
    data = {
        "embedding": emb_cpu,
        "original_size": original_size,
        "input_size": input_size,
        "model_version": SAM2_VERSION,
        "attr_name": used_attr,
    }
    os.makedirs(_embedding_group_dir(group), exist_ok=True)
    image_name = extract_image_name(image_path)
    torch.save(data, _embedding_file_path(group, image_name))
    return True, "ok"


def load_embedding(group: str, image_name: str):
    path = _embedding_file_path(group, image_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        state = torch.load(path, map_location=DEVICE)
        print(
            f"Loaded embedding for {image_name}: keys={list(state.keys()) if state else 'None'}"
        )
        return state
    except Exception as e:
        print(f"Error loading embedding from {path}: {e}")
        raise


def set_predictor_from_embedding(state: dict):
    if state is None:
        raise ValueError("State is None")
    if "embedding" not in state:
        raise ValueError(
            f"No 'embedding' key in state. Available keys: {list(state.keys())}"
        )

    emb = state["embedding"]
    if emb is None:
        raise ValueError("Embedding is None")

    attr_name = state.get("attr_name")

    def _to_device(obj):
        import torch as _torch

        if isinstance(obj, _torch.Tensor):
            return obj.to(DEVICE)
        if isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [_to_device(v) for v in obj]
            return type(obj)(t) if not isinstance(obj, list) else t
        return obj

    emb_dev = _to_device(emb)

    set_flag = False
    if attr_name and hasattr(predictor, attr_name):
        try:
            setattr(predictor, attr_name, emb_dev)
            set_flag = True
        except Exception:
            set_flag = False
    if not set_flag:
        for attr in [
            "_features",
            "features",
            "_image_embedding",
            "image_embedding",
            "_image_features",
        ]:
            if hasattr(predictor, attr):
                try:
                    setattr(predictor, attr, emb_dev)
                    set_flag = True
                    break
                except Exception:
                    continue
    if not set_flag:
        raise RuntimeError("No suitable attribute on predictor to set embedding")
    # Set size information that SAM2 expects
    if "original_size" in state:
        original_size = tuple(state["original_size"])
        if hasattr(predictor, "original_size"):
            predictor.original_size = original_size
        if hasattr(predictor, "_orig_hw"):
            predictor._orig_hw = [original_size]  # SAM2 expects a list
        if hasattr(predictor, "_orig_h") and hasattr(predictor, "_orig_w"):
            predictor._orig_h, predictor._orig_w = original_size

    if "input_size" in state:
        input_size = tuple(state["input_size"])
        if hasattr(predictor, "input_size"):
            predictor.input_size = input_size

    # Set flags to indicate image is set
    for flag_attr in ["_is_image_set", "is_image_set"]:
        if hasattr(predictor, flag_attr):
            try:
                setattr(predictor, flag_attr, True)
            except Exception:
                pass
            break


# Ensure embedding exists or create it using provided image


def ensure_embedding(group: str, image_name: str, image_rgb: np.ndarray = None):
    if embedding_exists(group, image_name):
        return True
    if image_rgb is None:
        return False
    predictor.set_image(image_rgb)
    emb = None
    used_attr = None
    for attr in [
        "_features",
        "features",
        "_image_embedding",
        "image_embedding",
        "_image_features",
    ]:
        if hasattr(predictor, attr):
            candidate = getattr(predictor, attr)
            if candidate is not None:
                emb = candidate
                used_attr = attr
                break
    if emb is None:
        return False

    def _to_cpu(obj):
        import torch as _torch

        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [_to_cpu(v) for v in obj]
            return type(obj)(t) if not isinstance(obj, list) else t
        return obj

    emb_cpu = _to_cpu(emb)
    original_size = getattr(predictor, "original_size", image_rgb.shape[:2])
    input_size = getattr(predictor, "input_size", image_rgb.shape[:2])
    data = {
        "embedding": emb_cpu,
        "original_size": original_size,
        "input_size": input_size,
        "model_version": SAM2_VERSION,
        "attr_name": used_attr,
    }
    os.makedirs(_embedding_group_dir(group), exist_ok=True)
    torch.save(data, _embedding_file_path(group, image_name))
    return True


# Cached prediction using precomputed embedding


def run_sam2_predict_cached(
    *,
    group: str,
    image_name: str,
    image_rgb: np.ndarray = None,
    points=None,
    labels=None,
    boxes=None,
    multimask=True,
) -> np.ndarray:
    if not group_precomputed(group) and image_rgb is None:
        raise RuntimeError(
            f"Group '{group}' not precomputed and no image provided to build embedding."
        )
    if not embedding_exists(group, image_name):
        if image_rgb is not None:
            created = ensure_embedding(group, image_name, image_rgb)
            if not created:
                raise FileNotFoundError(
                    f"Could not create embedding for '{image_name}' in '{group}'"
                )
        else:
            raise FileNotFoundError(
                f"Embedding for '{image_name}' missing and no image provided."
            )
    state = load_embedding(group, image_name)
    set_predictor_from_embedding(state)
    import torch as _torch

    kwargs = {}
    if points is not None:
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 2:
            pts = pts[None, :, :]
        kwargs["point_coords"] = _torch.tensor(pts, device=DEVICE)
    if labels is not None:
        lbls = np.array(labels, dtype=np.int64)
        if lbls.ndim == 1:
            lbls = lbls[None, :]
        kwargs["point_labels"] = _torch.tensor(lbls, device=DEVICE)
    if boxes is not None:
        b = np.array(boxes, dtype=np.float32)
        if b.shape == (2, 2):
            b = np.array([b[0, 0], b[0, 1], b[1, 0], b[1, 1]], dtype=np.float32)
        if b.ndim == 1:
            b = b[None, :]
        kwargs["box"] = _torch.tensor(b, device=DEVICE)
    kwargs["multimask_output"] = multimask
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(torch_dtype == torch.bfloat16)
    ):
        masks, scores, _ = predictor.predict(**kwargs)
    if masks is None:
        raise RuntimeError(
            "Predictor returned no masks (None). Causes: invalid/old embedding, image size mismatch, or unsupported prompt. Try re-precomputing this image embedding."
        )
    if not torch.is_tensor(scores):
        scores_t = torch.tensor(scores)
    else:
        scores_t = scores
    best_idx = int(torch.argmax(scores_t).item())
    if torch.is_tensor(masks):
        best = masks[best_idx].detach().cpu().numpy()
    else:
        best = np.array(masks)[best_idx]
    return (best > 0).astype(np.uint8) * 255
