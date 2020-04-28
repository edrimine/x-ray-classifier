import os
import pprint

import click
import numpy as np
import pkg_resources

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model  # noqa: E402
from tensorflow.keras.preprocessing import image  # noqa: E402


model_file = pkg_resources.resource_filename(
    "xray_classifier", "models/x_ray_vgg19_clf_v1.h5"
)
clf = load_model(model_file)

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
    "-f", "--file", prompt="File path", help="Full or relative filepath to the image"
)
@click.option(
    "-t",
    "--threshhold",
    prompt="Threshhold",
    default=0.5,
    type=float,
    help="Probability threshhold for classification",
)
@click.option(
    "--probs/--no-probs",
    default=False,
    help="Print probabilities for each class"
)
def classify(file, threshhold, probs):
    img = image.load_img(file, target_size=(244, 244))
    img_array = (np.array(img) * (1.0 / 255)).reshape((1, 244, 244, 3))
    pred = clf.predict(img_array)

    classes = pred[0] > threshhold
    print(
        f"Predicted classes with probability threshhold "
        f"of {threshhold}: {CLASS_NAMES[classes]}"
    )

    if probs:
        pprint.pprint(list(zip(CLASS_NAMES, pred[0])))


def main():
    classify()
