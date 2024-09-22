import pandas as pd
import os
from PIL import Image
import numpy as np
from keras import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


root_folder = "Disease_detection"
metadata_file = "MetaData.csv.xlsx"
metadata_df = pd.read_excel(metadata_file, sheet_name='Sheet1')

# Split the data into train, test, and validation sets
train_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Path to the normalized image folder
normalized_folder = "Disease_detection_normalized"
os.makedirs(normalized_folder, exist_ok=True)

# Load and process the images
normalized_images = []
max_width = 0
max_height = 0
for index, row in train_df.iterrows():
    image_name = row["Image_Name"].strip()  # Remove leading/trailing whitespace
    types = row["Type"].strip()  # Remove leading/trailing whitespace
    single_multiple = row["Single (S) / Multiple (M)"]
    disease_healthy = row["Disease/Healthy"].strip()  # Remove leading/trailing whitespace
    growth_stage = row["Growth_stage"]

    image_path = os.path.join(root_folder, disease_healthy, types, image_name + ".jpg")
    image = Image.open(image_path)

    # Update the maximum width and height
    width, height = image.size
    max_width = max(max_width, width)
    max_height = max(max_height, height)

    # Convert the image to numpy array
    image_array = np.array(image)

    # Append the image to the list
    normalized_images.append(image_array)

    # Print image details
    print("Image:", image_path)
    print("Disease Type:", types)
    print("Growth Stage:", growth_stage)
    print()

# Resize images to have the same dimensions
resized_images = []
for image in normalized_images:
    resized_image = np.resize(image, (max_height, max_width, 3))
    resized_images.append(resized_image)

# Save the normalized and resized images to the new folder
for i, image in enumerate(resized_images):
    image_name = train_df.iloc[i]["Image_Name"].strip()
    normalized_image_path = os.path.join(normalized_folder, image_name + ".jpg")
    Image.fromarray(image).save(normalized_image_path)

# Convert the images to numpy arrays
train_images = np.array(resized_images)

# Define the target variable (e.g., disease/healthy)
train_labels = np.array(train_df["Disease/Healthy"])

# Convert the target variable to one-hot encoding
num_classes = len(np.unique(train_labels))
train_labels = pd.get_dummies(train_labels).values

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(max_height, max_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)
