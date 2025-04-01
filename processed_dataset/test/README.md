To run the training files . run the command 
python train.py --folder1 processed_dataset --folder2 ProcessedDisfarDataset  --train_batches 70 --val_batches 18  --test_batches 14 --epochs 10

The ProcessedDisfarDataset contains the Processed Features from the Dataset in the folder order - test, train, val
each folder contains - eg: test_batch_0_features.npy , test_batch_0_images.npy, test_batch_0_labels.npy

The processed_dataset contains the feature extracted numpy array tensors in the folder order - test, train, val
each folder contains subfolder -  joint_features , labels_continous, labels_contagious where they are arranged in batch_0.npy and so on.

Note:
The Data cleansing and Feature extraction can be done for BOLD Dataset in the following code using Spatiotemporal variant cnn. (convo3d with lstm).
The file BolFeatureExtractionPipeline.py have been implemented this feature.

The Resnet50 feature extraction have been used for Disfar dataset and the features have been stored
The DisfarSkeleton.py and DisfarFeatureExtraction.py implements this features


The train.py implements both Feature level fusion and Decision level fusion .
