import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lw_cnn():
    """
    Builds and returns the full LW-CNN model and the feature extractor sub-model.
    Architecture:
    - Input: (128, 128, 1)
    - Conv2D: 32 filters, 3x3, ReLU, padding='same'
    - Conv2D: 64 filters, 3x3, ReLU, padding='same'
    - MaxPooling2D: 2x2
    - Flatten
    - Dense: 128 units, ReLU (Feature embedding layer)
    - Dropout: 0.5
    """
    inputs = layers.Input(shape=(128, 128, 1), name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    embedding = layers.Dense(128, activation='relu', name='embedding_layer')(x)
    outputs = layers.Dropout(0.5)(embedding)

    full_model = models.Model(inputs=inputs, outputs=outputs, name='LW_CNN_Full')
    feature_extractor = models.Model(inputs=inputs, outputs=embedding, name='LW_CNN_FeatureExtractor')

    return full_model, feature_extractor

# Global feature extractor model instance
_, _feature_extractor_model = build_lw_cnn()

def extract_image_features(image_path):
    """
    Loads an image from `image_path`, resizes and normalizes it to (128, 128, 1),
    and passes it through the feature extractor network to output a 128-d vector.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = tf.keras.utils.load_img(image_path, color_mode="grayscale", target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_tensor = tf.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 1)

    features = _feature_extractor_model.predict(img_tensor, verbose=0)
    return features[0]  # Shape: (128,)

if __name__ == "__main__":
    print("LW-CNN Feature Extractor initialized successfully.")
    sample_img = os.path.join("IR_fusion_images", os.listdir("IR_fusion_images")[0])
    vec = extract_image_features(sample_img)
    print(f"Sample feature vector extracted. Shape: {vec.shape}")
