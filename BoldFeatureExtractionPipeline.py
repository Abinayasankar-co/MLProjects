import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
from keras import layers, Model

class SpatioTemporalCNNLSTM(Model):
    def __init__(self, frame_height=18, frame_width=1, channels=1, frames_per_sequence=16, num_classes=26):
        super(SpatioTemporalCNNLSTM, self).__init__()
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.channels = channels
        self.frames_per_sequence = frames_per_sequence
        
        self.conv3d_1 = layers.Conv3D(64, (3, 3, 1), padding='same', activation='relu', dtype='float32')
        self.pool3d_1 = layers.MaxPooling3D((1, 2, 1), dtype='float32')
        self.conv3d_2 = layers.Conv3D(128, (3, 3, 1), padding='same', activation='relu', dtype='float32')
        self.pool3d_2 = layers.MaxPooling3D((1, 2, 1), dtype='float32')
        self.conv3d_3 = layers.Conv3D(256, (3, 3, 1), padding='same', activation='relu', dtype='float32')
        self.pool3d_3 = layers.MaxPooling3D((1, 2, 1), dtype='float32')
        
        self.projection = layers.TimeDistributed(layers.Dense(512, activation='relu'), dtype='float32')
        self.lstm_1 = layers.LSTM(256, return_sequences=True, dtype='float32')
        self.lstm_2 = layers.LSTM(128, dtype='float32')
        
        self.feature_layer = layers.Dense(128, activation='relu', dtype='float32')
        self.classifier = layers.Dense(num_classes, activation='sigmoid', dtype='float32')

    def build(self, input_shape):
        super(SpatioTemporalCNNLSTM, self).build(input_shape)
        
    def call(self, inputs, training=False):
        # Validate input shape
        expected_shape = (None, self.frames_per_sequence, self.frame_height, self.frame_width, self.channels)
        inputs = tf.ensure_shape(inputs, expected_shape)
        print(f"Input shape to SpatioTemporalCNNLSTM: {inputs.shape}")

        x = tf.cast(inputs, tf.float32)
        x = self.conv3d_1(x)
        print(f"After conv3d_1: {x.shape}")
        x = self.pool3d_1(x)
        print(f"After pool3d_1: {x.shape}")
        x = self.conv3d_2(x)
        print(f"After conv3d_2: {x.shape}")
        x = self.pool3d_2(x)
        print(f"After pool3d_2: {x.shape}")
        x = self.conv3d_3(x)
        print(f"After conv3d_3: {x.shape}")
        x = self.pool3d_3(x)
        print(f"After pool3d_3: {x.shape}")
        
        batch_size = tf.shape(x)[0]
        frames = tf.shape(x)[1]
        height = tf.shape(x)[2]
        width = tf.shape(x)[3]
        channels = tf.shape(x)[4]
        new_dim = height * width * channels
        x = tf.reshape(x, (batch_size, frames, new_dim))
        print(f"After reshape: {x.shape}")
        
        x = self.projection(x)
        print(f"After projection: {x.shape}")
        x = self.lstm_1(x)
        print(f"After lstm_1: {x.shape}")
        x = self.lstm_2(x)
        print(f"After lstm_2: {x.shape}")
        
        features = self.feature_layer(x)
        print(f"After feature_layer: {features.shape}")
        logits = self.classifier(features)
        print(f"After classifier: {logits.shape}")
        return features, logits

class FeatureExtractor:
    def __init__(self, batch_size=32, cache_dir="retrained_features", sub_batch_size=8):
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.cache_dir = cache_dir
        os.makedirs(os.path.join(cache_dir, "train", "joint_features"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "val", "joint_features"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "test", "joint_features"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "train", "label_categorical"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "val", "label_categorical"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "test", "label_categorical"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "train", "label_continuous"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "val", "label_continuous"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "test", "label_continuous"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "model_weights"), exist_ok=True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        else:
            print("No GPU found, using CPU")
        self.model = SpatioTemporalCNNLSTM(frame_height=18, frame_width=1, channels=1, frames_per_sequence=16, num_classes=26)
        self.model.build(input_shape=(None, 16, 18, 1, 1))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def _preprocess_joints(self, joints):
        # Input shape: (batch_size, 18, 16, 1) - (batch_size, height=joints, frames, channels)
        print(f"Before preprocessing joints shape: {joints.shape}")
        
        # Transpose to (batch_size, frames=16, height=18, channels=1)
        joints = tf.transpose(joints, perm=[0, 2, 1, 3])  # (batch_size, 16, 18, 1)
        
        # Add width dimension to get (batch_size, 16, 18, 1, 1)
        joints = tf.expand_dims(joints, axis=3)  # (batch_size, 16, 18, 1, 1)
        
        # Ensure shape
        joints = tf.ensure_shape(joints, (None, 16, 18, 1, 1))
        print(f"After preprocessing joints shape: {joints.shape}")
        return joints

    def _process_batch(self, batch_path, batch_idx, split_dir, categorical=None, continuous=None):
        cache_path = os.path.join(self.cache_dir, split_dir, "joint_features", f"batch_{batch_idx}.npy")
        cat_cache_path = os.path.join(self.cache_dir, split_dir, "label_categorical", f"batch_{batch_idx}.npy")
        cont_cache_path = os.path.join(self.cache_dir, split_dir, "label_continuous", f"batch_{batch_idx}.npy")
        if os.path.exists(cache_path) and os.path.exists(cat_cache_path) and os.path.exists(cont_cache_path):
            print(f"Loading cached features from {cache_path}")
            return np.load(cache_path), np.load(cat_cache_path), np.load(cont_cache_path)
        joints = np.load(batch_path)
        joints = tf.convert_to_tensor(joints, dtype=tf.float32)
        joints = self._preprocess_joints(joints)
        features = []
        for start in range(0, joints.shape[0], self.sub_batch_size):
            end = min(start + self.sub_batch_size, joints.shape[0])
            sub_batch = joints[start:end]
            sub_features, _ = self.model(sub_batch, training=False)
            features.append(sub_features.numpy())
        features = np.concatenate(features, axis=0)
        np.save(cache_path, features)
        if categorical is not None:
            np.save(cat_cache_path, categorical)
        if continuous is not None:
            np.save(cont_cache_path, continuous)
        print(f"Saved features to {cache_path}")
        return features, categorical, continuous

    def extract_features(self, dataset_dir="processed_dataset"):
        train_dir = os.path.join(dataset_dir, "train", "joint_features")
        val_dir = os.path.join(dataset_dir, "val", "joint_features")
        test_dir = os.path.join(dataset_dir, "test", "joint_features")

        def process_split(split_dir, split_name):
            if not os.path.exists(split_dir):
                print(f"Directory {split_dir} not found. Skipping {split_name} split.")
                return np.array([]), np.array([]), np.array([])
            joint_files = sorted([f for f in os.listdir(split_dir) if f.startswith("batch_") and f.endswith(".npy")])
            if not joint_files:
                print(f"No batch files found in {split_dir}. Skipping {split_name} split.")
                return np.array([]), np.array([]), np.array([])
            features, categorical_list, continuous_list = [], [], []
            for i, f in enumerate(joint_files):
                cat_path = os.path.join(dataset_dir, split_name, "labels_categorical", f"batch_{i}.npy")
                cont_path = os.path.join(dataset_dir, split_name, "labels_continuous", f"batch_{i}.npy")
                categorical = np.load(cat_path) if os.path.exists(cat_path) else None
                continuous = np.load(cont_path) if os.path.exists(cont_path) else None
                batch_features, batch_cat, batch_cont = self._process_batch(os.path.join(split_dir, f), i, split_name, categorical, continuous)
                features.append(batch_features)
                if batch_cat is not None:
                    categorical_list.append(batch_cat)
                if batch_cont is not None:
                    continuous_list.append(batch_cont)
            features = np.concatenate(features, axis=0) if features else np.array([])
            categorical = np.concatenate(categorical_list, axis=0) if categorical_list else np.array([])
            continuous = np.concatenate(continuous_list, axis=0) if continuous_list else np.array([])
            return features, categorical, continuous

        train_features, train_cat, train_cont = process_split(train_dir, "train")
        val_features, val_cat, val_cont = process_split(val_dir, "val")
        test_features, test_cat, test_cont = process_split(test_dir, "test")

        if train_features.size > 0:
            np.save(os.path.join(self.cache_dir, "train", "train_features.npy"), train_features)
        if val_features.size > 0:
            np.save(os.path.join(self.cache_dir, "val", "val_features.npy"), val_features)
        if test_features.size > 0:
            np.save(os.path.join(self.cache_dir, "test", "test_features.npy"), test_features)
        
        return (train_features, train_cat, train_cont), (val_features, val_cat, val_cont), (test_features, test_cat, test_cont)

class SkeletonDataset:
    def __init__(self, mode, normalize=True, use_xy=False, batch_size=32, frames_per_sequence=16, pretrained_weights_dir="pretrainardweights"):
        self.bold_path = "M:/FreelancerDataset/BOLD_public"
        self.mode = mode
        self.test_mode = (mode == 'test')
        self.normalize = normalize
        self.use_xy = use_xy
        self.batch_size = batch_size
        self.frames_per_sequence = frames_per_sequence
        self.num_joints = 18
        self.pretrained_weights_dir = pretrained_weights_dir
        self.categorical_emotions = [
            "Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence",
            "Happiness", "Pleasure", "Excitement", "Surprise", "Sympathy", 
            "Doubt/Confusion", "Disconnect", "Fatigue", "Embarrassment", "Yearning",
            "Disapproval", "Aversion", "Annoyance", "Anger", "Sensitivity", 
            "Sadness", "Disquietment", "Fear", "Pain", "Suffering"
        ]
        self.continuous_emotions = ["Valence", "Arousal", "Dominance"]
        csv_path = os.path.join(self.bold_path, "annotations", "test_meta.csv" if self.test_mode else f"{mode}.csv")
        self.df = pd.read_csv(csv_path, names=["video", "person_id", "min_frame", "max_frame"] + 
                              self.categorical_emotions + self.continuous_emotions + ["Gender", "Age", "Ethnicity", "annotation_confidence"])
        self.df["joints_path"] = self.df["video"].str.replace(r"(\d+)\.(mp4|npy)$", r"\1.npy", regex=True)
        self.max_x, self.max_y = self._load_pretrained_normalization()

    def _load_pretrained_normalization(self):
        max_x_path = os.path.join(self.pretrained_weights_dir, f"BOLD_{self.mode}_max_x_joint.npy")
        max_y_path = os.path.join(self.pretrained_weights_dir, f"BOLD_{self.mode}_max_y_joint.npy")
        if not os.path.exists(max_x_path) or not os.path.exists(max_y_path):
            print(f"Warning: Pretrained normalization files not found for {self.mode} split. Using default values.")
            return 1.0, 1.0
        max_x = np.load(max_x_path)
        max_y = np.load(max_y_path)
        print(f"Loaded pretrained max_x: {max_x}, max_y: {max_y} for {self.mode} split")
        return max_x.item() if max_x.size == 1 else max_x, max_y.item() if max_y.size == 1 else max_y

    def _generator(self):
        for index in range(len(self.df)):
            sample = self.df.iloc[index]
            joints = self._load_joints(sample)
            if joints is None:
                continue
            if not self.test_mode:
                categorical = sample[self.categorical_emotions].astype(np.float32).values
                continuous = sample[self.continuous_emotions].astype(np.float32).values / 10.0
                yield joints, categorical, continuous
            else:
                yield joints

    def _load_joints(self, sample):
        joints_path = os.path.normpath(os.path.join(self.bold_path, "joints", sample["joints_path"]))
        if not os.path.exists(joints_path):
            print(f"Warning: Joints file not found at {joints_path}")
            return None
        joints_data = np.load(joints_path)
        frames, features = joints_data.shape
        if features < 36:
            print(f"Warning: Insufficient features ({features}) in {joints_path}. Expected at least 36.")
            return None
        joints18 = joints_data[:, :36].reshape(frames, self.num_joints, 2)
        joints18[:, :, 0] -= joints18[0, :, 0]
        if self.normalize:
            joints18[:, :, 0] /= (self.max_x or 1.0)
            joints18[:, :, 1] /= (self.max_y or 1.0)
        if frames < self.frames_per_sequence:
            padding = np.zeros((self.frames_per_sequence - frames, self.num_joints, 2))
            joints18 = np.concatenate([joints18, padding], axis=0)
        elif frames > self.frames_per_sequence:
            joints18 = joints18[:self.frames_per_sequence, :, :]
        joints18 = np.transpose(joints18, (1, 0, 2))  # (joints=18, frames=16, features=1 or 2)
        if not self.use_xy:
            joints18 = joints18[:, :, 0:1]
        expected_shape = (self.num_joints, self.frames_per_sequence, 1 if not self.use_xy else 2)
        if joints18.shape != expected_shape:
            print(f"Skipping sample with incorrect shape: {joints18.shape}, expected: {expected_shape}")
            return None
        return joints18.astype(np.float32)

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(18, self.frames_per_sequence, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(26,), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32)
            ) if not self.test_mode else tf.TensorSpec(shape=(18, self.frames_per_sequence, 1), dtype=tf.float32)
        ).batch(
            self.batch_size,
            drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE)

    def save_dataset(self, output_dir="processed_dataset", train_split=0.8, val_split=0.1):
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        test_dir = os.path.join(output_dir, "test")
        
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, "joint_features"), exist_ok=True)
            if not self.test_mode:
                os.makedirs(os.path.join(split_dir, "labels_categorical"), exist_ok=True)
                os.makedirs(os.path.join(split_dir, "labels_continuous"), exist_ok=True)
        
        dataset = self.get_dataset()
        total_samples = len(self.df)
        train_samples = int(total_samples * train_split)
        val_samples = int(total_samples * val_split)
        train_batches = (train_samples + self.batch_size - 1) // self.batch_size
        val_batches = (val_samples + self.batch_size - 1) // self.batch_size
        
        for i, batch in enumerate(dataset):
            if i < train_batches:
                save_dir = train_dir
                batch_idx = i
            elif i < train_batches + val_batches:
                save_dir = val_dir
                batch_idx = i - train_batches
            else:
                save_dir = test_dir
                batch_idx = i - train_batches - val_batches
            joints, categorical, continuous = batch
            np.save(os.path.join(save_dir, "joint_features", f"batch_{batch_idx}.npy"), joints.numpy())
            if not self.test_mode:
                np.save(os.path.join(save_dir, "labels_categorical", f"batch_{batch_idx}.npy"), categorical.numpy())
                np.save(os.path.join(save_dir, "labels_continuous", f"batch_{batch_idx}.npy"), continuous.numpy())
            print(f"Saved batch {batch_idx} to {save_dir}")

def main(args):
    # Load and save dataset with pretrained normalization
    dataset = SkeletonDataset(
        mode="train",
        normalize=True,
        use_xy=False,
        batch_size=args.batch_size,
        frames_per_sequence=16,
        pretrained_weights_dir=args.pretrained_weights_dir
    )
    dataset.save_dataset(output_dir=args.output_dir, train_split=args.train_split, val_split=args.val_split)
    
    # Initialize feature extractor and retrain
    extractor = FeatureExtractor(batch_size=args.batch_size, sub_batch_size=args.sub_batch_size, cache_dir=args.cache_dir)
    
    # Prepare datasets for retraining
    train_dataset = dataset.get_dataset()
    val_dataset = SkeletonDataset(
        mode="val",
        normalize=True,
        use_xy=False,
        batch_size=args.batch_size,
        frames_per_sequence=16,
        pretrained_weights_dir=args.pretrained_weights_dir
    ).get_dataset()

    
    # Extract features after retraining
    (train_features, train_categorical, train_continuous), (val_features, val_categorical, val_continuous), (test_features, test_categorical, test_continuous) = extractor.extract_features(dataset_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sub_batch_size", type=int, default=4)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="processed_dataset")
    parser.add_argument("--cache_dir", type=str, default="retrained_features")
    parser.add_argument("--pretrained_weights_dir", type=str, default="pretrainardweights")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)