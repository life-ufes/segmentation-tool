import hashlib
import os
from glob import glob
from tqdm import tqdm
import pickle
import imagehash
from PIL import Image
import json


PAD_UFES_20_IMAGES_PATH = "/home/patcha/Datasets/PAD-UFES-20/images"
SEG_IMAGES_PATH = "/home/patcha/Datasets/pad__segmentation/all-data-png"
HASHS_PAD_PATH = "pad_hashs.pkl"
HASHS_SEG_PATH = "seg_hashs.pkl"



def _compute_hash_image_old(image_path, block_size=65536):
    """
    This function computes the hash of an image file.
    :param image_path: str
        The full path to the image file.
    :param block_size: int, optional (default=65536)
        The block size to read the file.
    :return: str        
    """
    
    sha = hashlib.sha256()
    
    with open(image_path, 'rb') as f:
        while True:
            bloco = f.read(block_size)
            if not bloco:
                break
            sha.update(bloco)

    return sha.hexdigest()


def _compute_hash_image(image_path):
    """
    This function computes the hash of an image file.
    :param image_path: str
        The full path to the image file.
    :param block_size: int, optional (default=65536)
        The block size to read the file.
    :return: str        
    """
    
    img = Image.open(image_path)
    return imagehash.average_hash(img)




def compute_images_hashes(FOLDER_PATH, img_ext="png"):
    """
    This function computes the hashes of the images in the PAD-UFES-20 dataset.
    :return: dict
        A dictionary with the hashes of the images in the PAD-UFES-20 dataset.
    """
    
    hashes = dict()
    images_path = glob(os.path.join(FOLDER_PATH, f"*.{img_ext}"))
    print(f"Number of images to compute the hash: {len(images_path)}")
    
    for img_path in tqdm(images_path, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        img_id = img_path.split(os.path.sep)[-1]
        hash_ = _compute_hash_image(img_path)
        hashes[img_id] = hash_

    return hashes


def compare_hashes(d1_hash_dict, d2_hash_dict, tol=10):
    """
    This function compares the hashes of two datasets.
    :param d1_hash_dict: dict
        A dictionary with the hashes of the images in the dataset 1.
    :param d2_hash_dict: dict
        A dictionary with the hashes of the images in the dataset 2.
    :return: dict
        A dictionary sharing the image ids that are in both datasets.
    """
        
    shared_hashes = dict()
    
    for d1_img_id, d1_hash in tqdm(d1_hash_dict.items(), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):           
        for d2_img_id, d2_hash in d2_hash_dict.items():                        
            similarity = abs(d1_hash - d2_hash) # hammign distance
            if similarity <= tol:                
                shared_hashes[d1_img_id] = d2_img_id
                break

    return shared_hashes


if __name__ == "__main__":
    
    try:
        with open(HASHS_PAD_PATH, 'rb') as f:
            pad_hashs = pickle.load(f)
        print("Hashs of PAD-UFES-20 images already computed.")
    except:
        pad_hashs = compute_images_hashes(PAD_UFES_20_IMAGES_PATH, img_ext="png")
        with open(HASHS_PAD_PATH, 'wb') as f:
            pickle.dump(pad_hashs, f)
        print("Hashs of PAD-UFES-20 images computed and saved.")

    try:
        with open(HASHS_SEG_PATH, 'rb') as f:
            seg_hashs = pickle.load(f)
        print("Hashs of SEG images already computed.")
    except:
        seg_hashs = compute_images_hashes(SEG_IMAGES_PATH, img_ext="png")
        with open(HASHS_SEG_PATH, 'wb') as f:
            pickle.dump(seg_hashs, f)
        print("Hashs of SEG images computed and saved.")


    shared_hashes = compare_hashes(seg_hashs, pad_hashs)
    print(f"Number of shared images: {len(shared_hashes)}")

    with open("shared_hashes.json", 'w') as f:
        json.dump(shared_hashes, f)

    # print(seg_hashs["c28a8dc9-1e08-4038-8185-f30a69efa719.png"])
    # print(pad_hashs["PAT_995_1867_165.png"])
    




    