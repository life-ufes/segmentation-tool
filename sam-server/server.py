from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import cv2
import os

# App routes remain here; heavy SAM logic moved to sam_utils
from sam_utils import (
    SESSIONS_FOLDER,
    MAIN_IMAGE_FOLDER,
    EMBEDDINGS_DIR,
    apply_mask_to_rgb,
    run_sam2_predict,
    run_sam2_predict_cached,
    embedding_exists,
    group_precomputed,
    compute_and_save_embedding,
    save_group_done,
    _embedding_group_dir,
    np,
    _precompute_lock,
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# SAM utils (model, predictor, embeddings) moved to sam_utils.py


# Update predict_box to pass rgb
@app.route("/predict/box", methods=["POST"])
def predict_box():
    data = request.json
    for key in ["fileName", "box", "folderName", "file"]:
        if key not in data:
            return jsonify({"error": f"Missing {key}"}), 400
    folderName = data["folderName"]
    sessionIdentifier = data.get("sessionIdentifier", "default")
    sessionFolder = os.path.join(SESSIONS_FOLDER, sessionIdentifier)
    os.makedirs(sessionFolder, exist_ok=True)
    originalFolder = os.path.join(sessionFolder, "original")
    maskedFolder = os.path.join(sessionFolder, "masked")
    os.makedirs(originalFolder, exist_ok=True)
    os.makedirs(maskedFolder, exist_ok=True)
    image = Image.open(io.BytesIO(base64.b64decode(data["file"])))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = image_cv2.shape[:2]
    image_name = data["fileName"]
    cv2.imwrite(os.path.join(originalFolder, image_name + ".jpg"), image_cv2)
    box = data["box"]
    if len(box) != 4:
        return jsonify({"error": "Box must be [x0,y0,x1,y1]"}), 400
    x0, y0, x1, y1 = box
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(w - 1, int(round(x0))))
    x1 = max(0, min(w - 1, int(round(x1))))
    y0 = max(0, min(h - 1, int(round(y0))))
    y1 = max(0, min(h - 1, int(round(y1))))
    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return jsonify({"error": "Box too small after clamping"}), 400
    clamped_box = [x0, y0, x1, y1]
    rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    try:
        mask_uint8 = run_sam2_predict_cached(
            group=folderName, image_name=image_name, image_rgb=rgb, boxes=clamped_box
        )
    except Exception as e:
        # Provide detailed context for debugging
        import traceback

        error_traceback = traceback.format_exc()
        print(f"Error in predict_box: {error_traceback}")
        return (
            jsonify(
                {
                    "error": str(e),
                    "traceback": error_traceback,
                    "context": {
                        "group": folderName,
                        "image": image_name,
                        "box": clamped_box,
                        "embedding_exists": embedding_exists(folderName, image_name),
                        "group_precomputed": group_precomputed(folderName),
                    },
                }
            ),
            500,
        )
    image_masked = apply_mask_to_rgb(mask_uint8)
    cv2.imwrite(os.path.join(maskedFolder, image_name + ".jpg"), image_masked)
    img_pil = Image.fromarray(image_masked)
    bio = io.BytesIO()
    img_pil.save(bio, format="PNG")
    bio.seek(0)
    return jsonify({"image": base64.b64encode(bio.getvalue()).decode()})


# Update predict_prompt similarly
@app.route("/predict/prompt", methods=["POST"])
def predict_prompt():
    data = request.json
    for key in ["fileName", "input_labels", "input_points", "folderName", "file"]:
        if key not in data:
            return jsonify({"error": f"Missing {key}"}), 400
    folderName = data["folderName"]
    sessionIdentifier = data.get("sessionIdentifier", "default")
    sessionFolder = os.path.join(SESSIONS_FOLDER, sessionIdentifier)
    os.makedirs(sessionFolder, exist_ok=True)
    originalFolder = os.path.join(sessionFolder, "original")
    maskedFolder = os.path.join(sessionFolder, "masked")
    os.makedirs(originalFolder, exist_ok=True)
    os.makedirs(maskedFolder, exist_ok=True)
    image = Image.open(io.BytesIO(base64.b64decode(data["file"])))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_name = data["fileName"]
    cv2.imwrite(os.path.join(originalFolder, image_name + ".jpg"), image_cv2)
    rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    try:
        mask_uint8 = run_sam2_predict_cached(
            group=folderName,
            image_name=image_name,
            image_rgb=rgb,
            points=data["input_points"],
            labels=data["input_labels"],
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    image_masked = apply_mask_to_rgb(mask_uint8)
    cv2.imwrite(os.path.join(maskedFolder, image_name + ".jpg"), image_masked)
    img_pil = Image.fromarray(image_masked)
    bio = io.BytesIO()
    img_pil.save(bio, format="PNG")
    bio.seek(0)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"].read()
    image = Image.open(io.BytesIO(file))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("image.jpg", image_cv2)

    h, w = image_cv2.shape[:2]
    center = [[w // 2, h // 2]]
    mask_uint8 = run_sam2_predict(
        image_rgb=cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB), points=center, labels=[1]
    )
    image_masked = apply_mask_to_rgb(mask_uint8)
    cv2.imwrite("image_masked.jpg", image_masked)

    image_masked = Image.fromarray(image_masked)

    image_io = io.BytesIO()
    image_masked.save(image_io, format="PNG")
    image_io.seek(0)
    base64_encoded_result = base64.b64encode(image_io.getvalue()).decode()
    return jsonify({"image": base64_encoded_result})


@app.route("/process/folder", methods=["GET"])
def process_folder():
    folder = request.args.get("folderName", None)

    if folder is None or folder == "":
        return jsonify({"error": "No folder in request"}), 400

    folder_path = os.path.join(MAIN_IMAGE_FOLDER, folder)

    if folder_path is None:
        return jsonify({"error": "No folder in request"}), 400

    # Check if folder exists
    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder does not exist"}), 400

    list_of_files = os.listdir(folder_path)
    if len(list_of_files) == 0:
        return jsonify({"error": "Folder is empty"}), 400
    return (
        jsonify({"message": "Folder validated (no preprocessing required for SAM2)."}),
        200,
    )


@app.route("/data/list", methods=["GET"])
def list_folders():
    list_of_folders = os.listdir(MAIN_IMAGE_FOLDER)
    return jsonify({"folders": list_of_folders}), 200


# New endpoints for listing and fetching images (expected by frontend)
@app.route("/image/list", methods=["GET"])
def image_list():
    folder = request.args.get("folderName")
    if not folder:
        return jsonify({"error": "No folderName in request"}), 400
    folder_path = os.path.join(MAIN_IMAGE_FOLDER, folder)
    if not os.path.isdir(folder_path):
        return jsonify({"error": "Folder does not exist"}), 404
    # Support jpg and png; return base names without extension
    images = []
    for f in os.listdir(folder_path):
        fl = f.lower()
        if fl.endswith(".jpg") or fl.endswith(".jpeg") or fl.endswith(".png"):
            images.append(os.path.splitext(f)[0])
    images.sort()
    return jsonify({"images": images, "count": len(images)}), 200


@app.route("/image", methods=["GET"])
def get_image():
    image_name = request.args.get("imageName")
    folder = request.args.get("folderName")
    if not image_name or not folder:
        return jsonify({"error": "Missing imageName or folderName"}), 400
    folder_path = os.path.join(MAIN_IMAGE_FOLDER, folder)
    if not os.path.isdir(folder_path):
        return jsonify({"error": "Folder does not exist"}), 404
    # Try jpg, jpeg, png
    candidates = [f"{image_name}.jpg", f"{image_name}.jpeg", f"{image_name}.png"]
    file_path = None
    for c in candidates:
        p = os.path.join(folder_path, c)
        if os.path.exists(p):
            file_path = p
            break
    if file_path is None:
        return jsonify({"error": "Image not found"}), 404
    img = cv2.imread(file_path)
    if img is None:
        return jsonify({"error": "Failed to read image"}), 500
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Store original dimensions for coordinate mapping
    original_height, original_width = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb)
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    bio.seek(0)
    return jsonify(
        {
            "image": base64.b64encode(bio.getvalue()).decode(),
            "name": image_name,
            "originalWidth": original_width,
            "originalHeight": original_height,
        }
    )


# JSON error handler for 404 (avoid HTML when frontend expects JSON)
@app.errorhandler(404)
def not_found(e):
    path = request.path
    if (
        path.startswith("/image")
        or path.startswith("/predict")
        or path.startswith("/precompute")
        or path.startswith("/data")
    ):
        return jsonify({"error": "Not found", "path": path}), 404
    return e


@app.route("/data/savetimer", methods=["POST"])
def save_timer():
    data = request.json
    if "sessionIdentifier" not in data:
        return jsonify({"error": "No sessionIdentifier in request"}), 400
    if "time" not in data:
        return jsonify({"error": "No time in request"}), 400
    sessionIdentifier = data["sessionIdentifier"]
    time = data["time"]
    os.makedirs(f"{SESSIONS_FOLDER}/{sessionIdentifier}", exist_ok=True)
    with open(f"{SESSIONS_FOLDER}/{sessionIdentifier}/time.txt", "w") as f:
        f.write(str(time))
    return jsonify({"message": "Time saved successfully"}), 200


@app.route("/precompute/folder", methods=["POST"])
def precompute_folder():
    data = request.json or {}
    folder = data.get("folderName")
    if not folder:
        return jsonify({"error": "No folderName in request"}), 400
    folder_path = os.path.join(MAIN_IMAGE_FOLDER, folder)
    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder does not exist"}), 400
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    if not image_files:
        return jsonify({"error": "Folder is empty"}), 400
    created = 0
    skipped = 0
    with _precompute_lock:
        for f in image_files:
            name_no_ext = os.path.splitext(f)[0]
            if embedding_exists(folder, name_no_ext):
                skipped += 1
                continue
            ok, msg = compute_and_save_embedding(folder, os.path.join(folder_path, f))
            if ok:
                created += 1
            else:
                return jsonify({"error": f"Failed embedding {f}: {msg}"}), 500
        save_group_done(folder)
    return (
        jsonify(
            {
                "message": "Precompute complete",
                "created": created,
                "skipped": skipped,
                "group": folder,
            }
        ),
        200,
    )


# After EMBEDDINGS_DIR creation add auto-scan
for grp in os.listdir(EMBEDDINGS_DIR):
    gpath = os.path.join(EMBEDDINGS_DIR, grp)
    if not os.path.isdir(gpath):
        continue
    has_pt = any(f.endswith(".pt") for f in os.listdir(gpath))
    done_flag = os.path.join(gpath, ".done")
    if has_pt and not os.path.exists(done_flag):
        with open(done_flag, "w") as f:
            f.write("ok")


@app.route("/precompute/status", methods=["GET"])
def precompute_status():
    folder = request.args.get("folderName")
    if not folder:
        return jsonify({"error": "No folderName in request"}), 400
    gdir = _embedding_group_dir(folder)
    if not os.path.isdir(gdir):
        return jsonify({"precomputed": False, "count": 0, "folder": folder}), 200
    files = [f for f in os.listdir(gdir) if f.endswith(".pt")]
    return (
        jsonify(
            {
                "precomputed": group_precomputed(folder),
                "count": len(files),
                "folder": folder,
            }
        ),
        200,
    )


@app.route("/")
def hello():
    return "Hello, World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
