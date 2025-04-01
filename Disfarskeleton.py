import os
import sys
import random 
import shutil
import numpy as np
from skimage import exposure, restoration, filters
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

class DisfarPreprocessing:
    def __init__(self):
       pass

    def preprocessing(self,image_tensor):
         try: 
            if image_tensor is None:
               return AttributeError('There is no image_tensor available')
            image_tensor = image_tensor.numpy()
            image_eq  = exposure.equalize_hist(image_tensor)
            image_wiener = restoration.wiener(image_eq,psf=np.ones((5, 5)) / 25, balance=0.1)
            image_gaussian = filters.gaussian(image_wiener,sigma=1.0)
            image_gabor, _ = filters.gabor(image_gaussian,frequency=0.6)
            image_laplacian = filters.laplace(image_gabor)
            return tf.reshape(image_laplacian,[-1])
         except Exception as e:
           print(e)

    def structure_dataset(self,input_dir: str, output_dir :str, training_ratio : float = 0.2):
         try:
          if input_dir is None:
             raise ValueError("Either Input Directory is structured Differently or it is a concern")
          
          if not output_dir or output_dir is None:
             raise ValueError("No scope of Valid output directory is found")
      
          train_dir = os.path.join(output_dir,"train")
          val_dir = os.path.join(output_dir,"val")

          if os.path.isdir(train_dir) and os.path.isdir(val_dir):
             print("Already Preprocessed")
             return
         
          if not train_dir and not val_dir:
            raise AssertionError("Training or validation directory is not formed")
          
          os.makedirs(train_dir,exist_ok=True)
          os.makedirs(val_dir,exist_ok=True)

          if not input_dir or input_dir is None:
             raise AttributeError("No scope of Valid Input Directory is mentioned")
          
          for class_name in os.listdir(input_dir):
             class_path = os.path.join(input_dir,class_name)
             if not os.path.isdir(class_path):
                continue
             
             images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path,f))]
             random.shuffle(images)

             split_index = int(len(images) * training_ratio)
             train_images = images[:split_index]
             val_images = images[split_index:]

             class_train_dir = os.path.join(train_dir,class_name)
             class_val_dir = os.path.join(val_dir,class_name)
             if not class_train_dir or not class_val_dir:
                raise AttributeError("Either of the classes of train and test have not been processed")
             
             os.makedirs(class_train_dir,exist_ok=True)
             os.makedirs(class_val_dir,exist_ok=True)

             
             for img_train in train_images:
                src = os.path.join(class_path,img_train)
                dest = os.path.join(class_train_dir,img_train)
                shutil.copy(src,dest)

             for img_val in val_images:
                src = os.path.join(class_path,img_val)
                dest = os.path.join(class_val_dir,img_val)
                shutil.copy(src,dest)

             print(f"Class '{class_name}' : {len(train_images)} Training images,{len(val_images)} Validation images")

         except Exception as e:
           print(f"{e}")

    def preprocess_and_save_dataset(self, base_path: str, output_dir: str, batch_size: int = 32, train_split=0.7, val_split=0.15):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        dataset_types = ['train', 'val']
        dataset_paths = [os.path.join(base_path, dt) for dt in dataset_types]

        if not os.path.exists(dataset_paths[0]):
            sys.exit(f"Train folder not found at {dataset_paths[0]}")
        class_names = sorted(os.listdir(dataset_paths[0]))
        label_map = {name: idx for idx, name in enumerate(class_names)}

        all_images = []
        all_labels = []

        for dataset_type, dataset_path in zip(dataset_types, dataset_paths):
            if not os.path.exists(dataset_path):
                print(f"Skipping {dataset_type} - path does not exist: {dataset_path}")
                continue

            for class_name in class_names:
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.isdir(class_path):
                    print(f"Skipping {class_name} in {dataset_type} - not a directory")
                    continue

                for image_file in os.listdir(class_path):
                    if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    image_path = os.path.join(class_path, image_file)
                    try:
                        image = Image.open(image_path).convert('L')
                        image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
                        processed_image = self.preprocessing(image_tensor)

                        if processed_image is not None:
                            all_images.append(processed_image.numpy())
                            all_labels.append(label_map[class_name])

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        continue

        all_images_np = np.array(all_images)
        all_labels_np = np.array(all_labels)

        if all_images_np.size == 0:
            print("No valid images processed")
            return

        total_samples = len(all_images_np)
        train_end = int(total_samples * train_split)
        val_end = train_end + int(total_samples * val_split)

        train_images = all_images_np[:train_end]
        val_images = all_images_np[train_end:val_end]
        test_images = all_images_np[val_end:]
        train_labels = all_labels_np[:train_end]
        val_labels = all_labels_np[train_end:val_end]
        test_labels = all_labels_np[val_end:]

        for split_name, images, labels in [('train', train_images, train_labels), 
                                           ('val', val_images, val_labels), 
                                           ('test', test_images, test_labels)]:
            output_subdir = os.path.join(output_dir, split_name)
            num_samples = len(images)
            num_batches = (num_samples + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_images = images[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]

                batch_file_prefix = f"{split_name}_batch_{batch_idx}"
                np.save(os.path.join(output_subdir, f"{batch_file_prefix}_images.npy"), batch_images)
                np.save(os.path.join(output_subdir, f"{batch_file_prefix}_labels.npy"), batch_labels)
                print(f"Saved {split_name} batch {batch_idx}: {batch_images.shape} images, {batch_labels.shape} labels")

        return all_images_np, all_labels_np, label_map

if __name__ == "__main__":
    input_path = r"M:\FreelancerDataset\DISFAR\FacialDataset"
    output_path = r"D:\MLProjects\ChristineClient\DisfarDataset"
    processed_dataset = r"D:\MLProjects\ChristineClient\ProcessedDisfarDataset"
    structuring_dataset = DisfarPreprocessing()
    structuring_dataset.preprocess_and_save_dataset(base_path=output_path, output_dir=processed_dataset)