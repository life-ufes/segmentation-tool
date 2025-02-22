

from flask import Flask, request, jsonify
from flask_cors import CORS  # import the flask_cors module
from PIL import Image
import io
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import cv2
import base64
import supervision as sv
import os 
from pathlib import Path

import pickle as pkl

SESSIONS_FOLDER = 'sessions'
MAIN_IMAGE_FOLDER = 'images'


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


os.makedirs(SESSIONS_FOLDER, exist_ok=True)

# remove any folder from the sessions folder that does not have any image inside the folder masked
for folder in os.listdir(SESSIONS_FOLDER):
    folder_path = os.path.join(SESSIONS_FOLDER, folder)
    masked_folder_path = os.path.join(folder_path, 'masked')
    if len(os.listdir(masked_folder_path)) == 0:
        rmdir(folder_path)

app = Flask(__name__)
CORS(app)  # enable CORS on the app


CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
#CHECKPOINT_PATH = "./sam_vit_b_01ec64.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
MODEL_TYPE = "vit_h"
#MODEL_TYPE = "vit_b"

model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
model.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model,
                                           crop_n_layers=1,
                                           crop_n_points_downscale_factor=2
                                           )

                                           
predictor = SamPredictor(model)

current_image = None
current_mask = None


def get_embedding(image_name, image_folder):
    # load pkl file
    with open(f'{image_folder}/{image_name}.pkl', 'rb') as f:
        embedding = pkl.load(f)
    return embedding


def get_largest_area(result_dict):
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

    # Overlay the mask and image
    # overlay_image = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)

    return mask_rgb


# def generate_image_of_mask(output_mask, shape):
#     image_rgb = np.zeros((shape[0], shape[1], 3), dtype='uint8')

#     for i in range(len(output_mask)):
#         mask = output_mask[i]['segmentation']
#         mask = np.where(mask, 255, 0).astype('uint8')
#         color = np.random.randint(0, 255, 3)
#         image = apply_mask(image_rgb, mask, color=color)
#         image_rgb = cv2.addWeighted(image_rgb, 1, image, 1, 0)

#     return image


def get_current_image():
    return current_image


def set_current_image(image):
    current_image = image



def generate_image(image):
    # Generate segmentation mask
    output_mask = mask_generator.generate(image)
    # get second largest area
    largest_area = get_largest_area(output_mask)
    mask = largest_area['segmentation']

    return mask


def set_image(image):
    predictor.set_image(image)


def generate_images_with_box(image, box):
    box = np.array(box)

    # exit()
    output_mask, scores, logits = predictor.predict(
                            box=box,
                            multimask_output=True,
                        )
    
    mask_input = output_mask[np.argmax(scores), :, :]  # Choose the model's best mask

    # return generate_image_of_mask(output_mask, image.shape)

    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(mask_input, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask, color=color)

    # for i in range(len(output_mask)):
    #     mask = output_mask[i]
    #     mask = np.where(mask, 255, 0).astype('uint8')
    
    #     color = (255, 255, 255)
    #     image = apply_mask(image_rgb, mask, color=color)

        #invert black and white
    # image = cv2.bitwise_not(image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def generate_image_with_prompt(image, input_labels, input_points):
    # input_labels = input_labels.split(',') if input_labels else []
    # input_points = input_points.split(',') if input_points else []
    input_points = np.array(input_points)
    # input_labels = np.array(input_labels, dtype=np.float32)
    print(input_points)
    print(input_labels)
    # exit()
    
    output_mask, scores, logits = predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True,
                        )
    
    mask_input = output_mask[np.argmax(scores), :, :]  # Choose the model's best mask


    # return generate_image_of_mask(output_mask, image.shape)

    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(mask_input, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask, color=color)

    # for i in range(len(output_mask)):
    #     mask = output_mask[i]
    #     mask = np.where(mask, 255, 0).astype('uint8')
    
    #     color = (255, 255, 255)
    #     image = apply_mask(image_rgb, mask, color=color)

        #invert black and white
    # image = cv2.bitwise_not(image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


@app.route('/save', methods=['POST'])
def save():
    data = request.json
    if 'sessionIdentifier' not in data:
        return jsonify({'error': 'No sessionIdentifier in request'}), 400
    if 'maskedImage' not in data:
        return jsonify({'error': 'No maskedImage in request'}), 400
    if 'fileName' not in data:
        return jsonify({'error': 'No fileName in request'}), 400
    sessionIdentifier = data['sessionIdentifier']
    sessionFolder = os.path.join(SESSIONS_FOLDER, sessionIdentifier)

    originalFolder = os.path.join(sessionFolder, 'original')
    maskedFolder = os.path.join(sessionFolder, 'masked')

    os.makedirs(originalFolder, exist_ok=True)
    os.makedirs(maskedFolder, exist_ok=True)

    image = Image.open(io.BytesIO(base64.b64decode(data['maskedImage']))
                        ).convert('RGB')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_name = data['fileName']
    cv2.imwrite(os.path.join(maskedFolder, image_name + '.jpg'), image_cv2)

    original_image = Image.open(io.BytesIO(base64.b64decode(data['originalImage']))).convert('RGB')
    original_image_cv2 = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(originalFolder, image_name + '.jpg'), original_image_cv2)
    
    return jsonify({'message': 'Saved successfully'}), 200

# list all image names on the camera_id folder
@app.route("/image/list", methods=['GET'])
def list_images():
    folder_name = request.args.get('folderName')
    if folder_name is None:
        return jsonify({'error': 'No folderName in request'}), 400

    folder_path = os.path.join(MAIN_IMAGE_FOLDER, folder_name)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder does not exist'}), 400
    list_of_files = os.listdir(folder_path)
    list_of_files = [file for file in list_of_files if file.endswith('.jpg')]
    list_of_files = [file.split('.')[0] for file in list_of_files]

    return jsonify({'images': list_of_files}), 200


@app.route("/image", methods=['GET'])
def get_image():
    folderName = request.args.get('folderName')
    image_name = request.args.get('imageName')
    if folderName is None:
        return jsonify({'error': 'No folderName in request'}), 400
    
    if image_name is None: 
        return jsonify({'error': 'No imageName in request'}), 400

    folder_path = os.path.join(MAIN_IMAGE_FOLDER, folderName)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder does not exist'}), 400

    image_path = os.path.join(folder_path, image_name)
    #add .jpg extension
    image_path = image_path + '.jpg'
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image does not exist'}), 400
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(image)
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    img_bytes = img_io.getvalue()
    base64_encoded_result = base64.b64encode(img_bytes).decode()
    return jsonify({'image': base64_encoded_result}), 200


@app.route('/predict/box', methods=['POST'])
def predict_box():
    data = request.json
    if 'fileName' not in data:
        return jsonify({'error': 'No file in request'}), 400
    if 'box' not in data:
        return jsonify({'error': 'No box in request'}), 400
    
    sessionIdentifier = data['sessionIdentifier']
    folderName = data['folderName']

    # check if a folder with the sessionIdentifier exists 
    sessionFolder = os.path.join(SESSIONS_FOLDER, sessionIdentifier)
    os.makedirs(sessionFolder, exist_ok=True)

    #create a original and masked folder
    originalFolder = os.path.join(sessionFolder, 'original')
    maskedFolder = os.path.join(sessionFolder, 'masked')
    os.makedirs(originalFolder, exist_ok=True)
    os.makedirs(maskedFolder, exist_ok=True)

    image = Image.open(io.BytesIO(base64.b64decode(data['file'])))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image_name = data['fileName'] 

    cv2.imwrite(os.path.join(originalFolder, image_name + '.jpg'), image_cv2)

    box = data['box']
    embeddingFolder = os.path.join(MAIN_IMAGE_FOLDER, folderName)

    if not os.path.exists(embeddingFolder):
        return jsonify({'error': 'Folder does not exist'}), 400
    
    predictor.reset_image()
    image_obj = get_embedding(image_name, embeddingFolder)
    predictor.features = image_obj['embedd']
    predictor.original_size = image_obj['original_size']
    #features is a torch tensor, set it to device 
    predictor.features = predictor.features.to(DEVICE)
    predictor.input_size = image_obj['input_size']
    predictor.is_image_set = True

    image_masked = generate_images_with_box(image_cv2, box)

    cv2.imwrite(os.path.join(maskedFolder, image_name + '.jpg'), image_masked)

    # turn black to white and white to black 
    # image_masked = cv2.bitwise_not(image_masked)

    image_masked = Image.fromarray(image_masked)

    #send image as response to the client in json format
    image = io.BytesIO()
    image_masked.save(image, format='PNG')
    image.seek(0)
    image_bytes = image.getvalue()
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

    sessionIdentifier = data['sessionIdentifier']
    folderName = data['folderName']

    # check if a folder with the sessionIdentifier exists 
    sessionFolder = os.path.join(SESSIONS_FOLDER, sessionIdentifier)
    os.makedirs(sessionFolder, exist_ok=True)

    #create a original and masked folder
    originalFolder = os.path.join(sessionFolder, 'original')
    maskedFolder = os.path.join(sessionFolder, 'masked')
    os.makedirs(originalFolder, exist_ok=True)
    os.makedirs(maskedFolder, exist_ok=True)

    image = Image.open(io.BytesIO(base64.b64decode(data['file'])))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image_name = data['fileName'] 

    cv2.imwrite(os.path.join(originalFolder, image_name + '.jpg'), image_cv2)

    input_labels = data['input_labels']
    input_points = data['input_points']

    embeddingFolder = os.path.join(MAIN_IMAGE_FOLDER, folderName)

    if not os.path.exists(embeddingFolder):
        return jsonify({'error': 'Folder does not exist'}), 400
    
    predictor.reset_image()
    image_obj = get_embedding(image_name, embeddingFolder)
    predictor.features = image_obj['embedd']
    predictor.original_size = image_obj['original_size']
    predictor.input_size = image_obj['input_size']
    predictor.is_image_set = True

    image_masked = generate_image_with_prompt(image_cv2, input_labels, input_points)

    cv2.imwrite(os.path.join(maskedFolder, image_name + '.jpg'), image_masked)

    image_masked = Image.fromarray(image_masked)

    image = io.BytesIO()
    image_masked.save(image, format='PNG')
    image.seek(0)
    image_bytes = image.getvalue()
    base64_encoded_result = base64.b64encode(image_bytes).decode()
    return jsonify({'image': base64_encoded_result})



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file'].read()  # get the file from the request
    image = Image.open(io.BytesIO(file))  # open the image
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # save image to disk
    cv2.imwrite('image.jpg', image_cv2)

    image_masked = generate_image(image_cv2)
    # save masked image to disk
    cv2.imwrite('image_masked.jpg', image_masked)
    
    image_masked = Image.fromarray(image_masked)

    # send image as response to the client in json format
    image = io.BytesIO()
    image_masked.save(image, format='PNG')
    image.seek(0)
    image_bytes = image.getvalue()
    base64_encoded_result = base64.b64encode(image_bytes).decode()  # encode as base64
    return jsonify({'image': base64_encoded_result})

@app.route('/process/folder', methods=['GET'])
def process_folder():
    # # Access 'folder' from the query parameters
    folder = request.args.get('folderName', None)

    if folder is None or folder == '':
        return jsonify({'error': 'No folder in request'}), 400

    folder_path = os.path.join(MAIN_IMAGE_FOLDER, folder)
    
    if folder_path is None:
        return jsonify({'error': 'No folder in request'}), 400

    # Check if folder exists
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder does not exist'}), 400
    
    try: 
        list_of_files = os.listdir(folder_path)
        # Check if folder is empty
        if len(list_of_files) == 0:
            return jsonify({'error': 'Folder is empty'}), 400
       
        for file in list_of_files:
            image = cv2.imread(os.path.join(folder_path, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Assuming predictor is defined elsewhere and set up correctly
            set_image(image)
            embedd = predictor.get_image_embedding()
            original_size = predictor.original_size
            input_size = predictor.input_size
            file = file.split('.')[0]
            predictor_obj = {'embedd': embedd, 'original_size': original_size,
                             'input_size': input_size}
            with open(f'{folder_path}/{file}.pkl', 'wb') as f:
                pkl.dump(predictor_obj, f)
    except Exception as e:
        return jsonify({'error': f'Error processing folder: {str(e)}'}), 400

    return jsonify({'message': 'Embeddings generated and serialized successfully'}), 200


# list the folders at images folder
@app.route('/data/list', methods=['GET'])
def list_folders():
    list_of_folders = os.listdir(MAIN_IMAGE_FOLDER)
    return jsonify({'folders': list_of_folders}), 200

@app.route('/data/savetimer', methods=['POST'])
def save_timer():
    data = request.json
    if 'sessionIdentifier' not in data:
        return jsonify({'error': 'No sessionIdentifier in request'}), 400
    if 'time' not in data:
        return jsonify({'error': 'No time in request'}), 400
    sessionIdentifier = data['sessionIdentifier']
    time = data['time']
    with open(f'{SESSIONS_FOLDER}/{sessionIdentifier}/time.txt', 'w') as f:
        f.write(str(time))
    return jsonify({'message': 'Time saved successfully'}), 200


@app.route('/')
def hello():
    return 'Hello, World!'



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
