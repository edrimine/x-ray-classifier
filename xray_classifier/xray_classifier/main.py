import os
import pprint
from pathlib import Path

import click
import numpy as np
import pkg_resources

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model_file = pkg_resources.resource_filename(
    "xray_classifier", "models/x_ray_vgg19_clf_v1.h5"
)

CLASS_NAMES = np.array(
    [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumothorax",
    ]
)


@click.command()
@click.option(
    "-t",
    "--threshold",
    default=0.5,
    type=float,
    help="Probability threshold for classification",
)
@click.option(
    "--probs/--no-probs", default=False, help="Print probabilities for each class"
)
@click.argument("filepath", type=click.Path(exists=True))
def classify(threshold, probs, filepath):
    """Predict FILEPATH classification."""

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image

    clf = load_model(model_file)

    if os.path.isdir(filepath):
        files = list(Path(filepath).glob("*.png"))
    else:
        files = [Path(filepath)]

    for file in files:
        img = image.load_img(file, target_size=(244, 244))
        img_array = (np.array(img) * (1.0 / 255)).reshape((1, 244, 244, 3))
        pred = clf.predict(img_array)

        classes = pred[0] > threshold
        print(file)
        print(
            f"Predicted classes with probability threshold "
            f"of {threshold}: {CLASS_NAMES[classes]}"
        )

        if probs:
            pprint.pprint(list(zip(CLASS_NAMES, pred[0])))


def main():
    classify()
