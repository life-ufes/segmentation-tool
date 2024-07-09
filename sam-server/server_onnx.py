

from flask import Flask, request, jsonify
from flask_cors import CORS  # import the flask_cors module
from PIL import Image
import io
import torch
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic


import numpy as np
import cv2
import base64

import warnings

app = Flask(__name__)
CORS(app)  # enable CORS on the app

CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
# CHECKPOINT_PATH = "./sam_vit_b_01ec64.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)


current_image = None
current_mask = None

image_embedding = None

onnx_model_path = "sam_onnx_quantized.onnx"

#check if onnx model exists
try:
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
except:
    print('Onnx model not found, creating it now...')
    import warnings

    onnx_model_not_quint_path = "sam_onnx_example.onnx"

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_not_quint_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    onnx_model_quantized_path = "sam_onnx_quantized.onnx"
    quantize_dynamic(
        model_input=onnx_model_not_quint_path,
        model_output=onnx_model_quantized_path,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )
    onnx_model_path = onnx_model_quantized_path
    ort_session = onnxruntime.InferenceSession(onnx_model_not_quint_path)


predictor = SamPredictor(sam)



def get_second_largest_area(result_dict):
    sorted_result = sorted(result_dict, key=(lambda x: x['area']),
                           reverse=True)
    return sorted_result[0]


def apply_mask(image, mask, color=None):
    print(image.shape, mask.shape)
    # Convert the mask to a 3 channel image
    if color is None:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask > 0] = color


    return mask_rgb



def get_current_image():
    return current_image

def set_current_image(image):
    current_image = imagese



def generate_image(image):
    # Generate segmentation mask
    output_mask = mask_generator.generate(image)
    # get second largest area
    largest_area = get_second_largest_area(output_mask)
    mask = largest_area['segmentation']

    return mask


def set_image(image):
    predictor.set_image(image)




@app.route('/predict/box', methods=['POST'])
def predict_box():
    data = request.json
    if 'file' not in data:
        return jsonify({'error': 'No file in request'}), 400
    if 'box' not in data:
        return jsonify({'error': 'No box in request'}), 400

    image = Image.open(io.BytesIO(base64.b64decode(data['file'])))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #resize image to [1500, 2250]
    # image_cv2 = cv2.resize(image_cv2, (1500, 2250), interpolation=cv2.INTER_AREA)
    #save image
    cv2.imwrite('image.png', image_cv2)

    box = data['box']
    print('box', box)

    onnx_box_coords = np.array(box).reshape(2, 2)
    onnx_box_labels = np.array([2,3])

    if (data['again'] is False or predictor.is_image_set is False):
        print('Setting image')
        set_image(image_cv2)
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    onnx_coord = np.concatenate([onnx_box_coords, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([onnx_box_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = predictor.transform.apply_coords(onnx_coord, image_cv2.shape[:2]).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image_cv2.shape[:2], dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    image_rgb = np.zeros((image_cv2.shape[0], image_cv2.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(masks, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask[0][0], color=color)

    # invert black and white
    # image = cv2.bitwise_not(image)

    #save image
    cv2.imwrite('image_masked.png', image)

    # send image as response to the client in json format
    image_by = io.BytesIO()
    image_masked = Image.fromarray(image)


    image_masked.save(image_by, format='PNG')
    image_by.seek(0)
    image_bytes = image_by.getvalue()
    base64_encoded_result = base64.b64encode(image_bytes).decode()
    return jsonify({'image': base64_encoded_result})


    

@app.route('/predict/prompt', methods=['POST'])
def predict_prompt():
    data = request.json
    if 'file' not in data:
        return jsonify({'error': 'No file in request'}), 400
    if 'input_labels' not in data:
        return jsonify({'error': 'No input_labels in request'}), 400
    if 'input_points' not in data:
        return jsonify({'error': 'No input_points in request'}), 400

    image = Image.open(io.BytesIO(base64.b64decode(data['file'])))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #resize image to [1500, 2250]
    # image_cv2 = cv2.resize(image_cv2, (1500, 2250), interpolation=cv2.INTER_AREA)
    #save image
    cv2.imwrite('image.png', image_cv2)

    input_labels = data['input_labels']
    input_points = data['input_points']
    # input_points = np.array([[712, 571]])

    print('input_labels', input_labels)
    print('input_points', input_points)


    if (data['again'] is False or predictor.is_image_set is False):
        print('Setting image')
        set_image(image_cv2)
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    # ort_session = onnxruntime.InferenceSession(onnx_model_path)
    onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = predictor.transform.apply_coords(onnx_coord, image_cv2.shape[:2]).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image_cv2.shape[:2], dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    image_rgb = np.zeros((image_cv2.shape[0], image_cv2.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(masks, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask[0][0], color=color)

    # invert black and white
    # image = cv2.bitwise_not(image)
    #save image
    cv2.imwrite('image_masked.png', image)

    # send image as response to the client in json format
    image_by = io.BytesIO()
    image_masked = Image.fromarray(image)

    image_masked.save(image_by, format='PNG')
    image_by.seek(0)
    image_bytes = image_by.getvalue()
    base64_encoded_result = base64.b64encode(image_bytes).decode()
    return jsonify({'image': base64_encoded_result})




@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file'].read()  # get the file from the request
    image = Image.open(io.BytesIO(file))  # open the image
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image_masked = generate_image(image_cv2)
    
    image_masked = Image.fromarray(image_masked)

    #send image as response to the client in json format
    image = io.BytesIO()
    image_masked.save(image, format='PNG')
    image.seek(0)
    image_bytes = image.getvalue()
    base64_encoded_result = base64.b64encode(image_bytes).decode()  # encode as base64
    return jsonify({'image': base64_encoded_result})
@app.route('/')
def hello():
    return 'Hello, World!'



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)