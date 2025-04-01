import os
import numpy as np
import argparse
import tensorflow as tf
import keras
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import LayerNormalization, Dense, Add, Input, Conv2D
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

class DisfarFeatureExtraction:
    def __init__(self, data_dir: str, num_threads: int, cache_size: int):
        self.data_dir = data_dir
        self.num_threads = num_threads
        self.cache_size = cache_size

        base_model = ResNet50(weights="imagenet", include_top=False, pooling=None)
        x = base_model.output
        x = LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)

        skip = base_model.get_layer('conv4_block6_out').output
        skip = LayerNormalization()(skip)
        skip = Conv2D(2048, (1, 1), padding='same')(skip)
        skip = keras.layers.GlobalAveragePooling2D()(skip)

        x = Add()([x, skip])
        x = Dense(512, activation='relu')(x)
        self.model = Model(inputs=base_model.input, outputs=x)

    def process_all_batches(self):
        for dataset_type in ["train", "val", "test"]:
            dataset_dir = os.path.join(self.data_dir, dataset_type)
            if not os.path.exists(dataset_dir):
                print(f"Skipping {dataset_type} - directory not found: {dataset_dir}")
                continue
            batch_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) 
                          if f.endswith("_images.npy")]
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                results = list(executor.map(self.process_batch_files, batch_files))
            print(f"Processed {dataset_type}: {len(results)} batches")

    @lru_cache(maxsize=1000)
    def process_batch_files(self, batch_file: str) -> str:
        try:
            images = np.load(batch_file)
            num_images = images.shape[0]
            print(f"Loaded {batch_file} with shape: {images.shape}")

            if len(images.shape) == 2:  
                total_pixels = images.shape[1]
                image_size = int(np.sqrt(total_pixels))
                if image_size * image_size != total_pixels:
                    raise ValueError(f"Cannot reshape {total_pixels} into a square image")
                images = images.reshape((num_images, image_size, image_size, 1))
                images = np.repeat(images, 3, axis=-1) 
            elif len(images.shape) == 3: 
                images = np.expand_dims(images, axis=-1)  
                images = np.repeat(images, 3, axis=-1)   
            elif len(images.shape) == 4:  
                if images.shape[-1] == 1:  
                    images = np.repeat(images, 3, axis=-1)  
                elif images.shape[-1] != 3:
                    raise ValueError(f"Unsupported channel count: {images.shape[-1]}, expected 1 or 3")

            images = tf.image.resize(images, (224, 224), method='bilinear')
            images = tf.keras.applications.resnet50.preprocess_input(images)

            features = self.model.predict(images, batch_size=32)

            feature_filename = batch_file.replace("_images.npy", "_features.npy")
            np.save(feature_filename, features)
            print(f"Saved features to {feature_filename} - shape: {features.shape}")
            return feature_filename

        except Exception as e:
            print(f"Error processing {batch_file}: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from .npy image batches using modified ResNet50.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing the file")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel threads for processing")
    parser.add_argument("--cache_size", type=int, default=32, help="Cache size for storing and faster processing")
    args = parser.parse_args()
    features_extracted = DisfarFeatureExtraction(
        data_dir=args.data_dir,
        num_threads=args.threads,
        cache_size=args.cache_size
    )
    features_extracted.process_all_batches()