from pathlib import Path
import numpy as np
import ntpath


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_classes(class_txt_file):
    with open(class_txt_file, 'r') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def add_image_id(model_outputs, image_id):
    model_outputs_updated = []
    if model_outputs != []:
        for output in model_outputs:
            output["image_id"] = image_id
            model_outputs_updated.append(output.copy())
    return model_outputs_updated

def sigmoid(x):
    return 1 / (1 + np.exp(-x))