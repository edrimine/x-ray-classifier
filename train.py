import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import Data_Entry_2017.csv file which has file names and labels for all images.
df = pd.read_csv("../data/Data_Entry_2017.csv")
df["labels"] = df["Finding Labels"].apply(lambda x: x.split("|"))
df.rename(columns={"Image Index": "filename", "Patient ID": "patient_id"}, inplace=True)
image_df = df[["filename", "patient_id", "labels"]]
image_df.head()

print(f"Total number of images {len(image_df)}")

# Split train/test data based on Patient ID. We want to ensure that there is not
# overlap in patients between the train and test sets as this would likely
# cause model testing metrics to appear better than they might actually be.
unique_patient_ids = list(image_df.patient_id.unique())
patient_id_train_full, patient_id_test = train_test_split(
    unique_patient_ids, test_size=0.25, random_state=333
)

# Further Break Traing Set into Train/Validate Set
patient_id_train, patient_id_validate = train_test_split(
    patient_id_train_full, test_size=0.1, random_state=222
)

train_df = image_df.query(f"patient_id in {patient_id_train}")
validate_df = image_df.query(f"patient_id in {patient_id_validate}")
test_df = image_df.query(f"patient_id in {patient_id_test}")

print(f"Train:\t\t  {len(train_df)}")
print(f"Validate:\t+ {len(validate_df)}")
print(f"Test:\t\t+ {len(test_df)}")
print(f"Total:\t\t= {len(image_df)}")

# Build Image Generators
train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
)
validate_image_generator = ImageDataGenerator(rescale=1.0 / 255)
test_image_generator = ImageDataGenerator(rescale=1.0 / 255)

train_data_gen = train_image_generator.flow_from_dataframe(
    train_df,
    directory="../data/images",
    x_col="filename",
    y_col="labels",
    color_mode="rgb",
)

validate_data_gen = validate_image_generator.flow_from_dataframe(
    validate_df,
    directory="../data/images",
    x_col="filename",
    y_col="labels",
    color_mode="rgb",
)

test_data_gen = test_image_generator.flow_from_dataframe(
    test_df,
    directory="../data/images",
    x_col="filename",
    y_col="labels",
    color_mode="rgb",
)

# Build Model
x_ray_clf = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.Dense(15, activation="sigmoid"),
    ]
)

x_ray_clf.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

history = x_ray_clf.fit_generator(
    train_data_gen,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validate_data_gen,
    validation_steps=50,
)

# save the classifier for future use
x_ray_clf.save("x_ray_clf_v1.h5")
