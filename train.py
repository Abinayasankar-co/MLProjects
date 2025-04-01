import os
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import pickle
import argparse
import warnings
import logging
from sklearn.base import clone

warnings.filterwarnings('ignore')

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def load_batch_data_folder1(folder, batch_idx, dataset_type):
    try:
        base_path = os.path.join(folder, dataset_type)
        feature_file = os.path.join(base_path, "joint_features", f"batch_{batch_idx}.npy")
        category_file = os.path.join(base_path, "labels_categorical", f"batch_{batch_idx}.npy")
        if not all(os.path.exists(f) for f in [feature_file, category_file]):
            return None, None
        features = np.load(feature_file)
        categories = np.load(category_file)
        if features.shape[0] != categories.shape[0]:
            print(f"Shape mismatch in {dataset_type} batch {batch_idx}")
            return None, None
        return features, categories
    except Exception as e:
        print(f"Error loading folder1 batch {batch_idx} ({dataset_type}): {e}")
        return None, None

def load_batch_data_folder2(folder, batch_idx, dataset_type):
    try:
        base_path = os.path.join(folder, dataset_type)
        feature_file = os.path.join(base_path, f"{dataset_type}_batch_{batch_idx}_features.npy")
        label_file = os.path.join(base_path, f"{dataset_type}_batch_{batch_idx}_labels.npy")
        if not all(os.path.exists(f) for f in [feature_file, label_file]):
            return None, None
        return np.load(feature_file), np.load(label_file)
    except Exception as e:
        print(f"Error loading folder2 batch {batch_idx} ({dataset_type}): {e}")
        return None, None

def normalize_features(features):
    try:
        scaler = RobustScaler()
        return scaler.fit_transform(features.reshape(features.shape[0], -1))
    except Exception as e:
        print(f"Normalization error: {e}")
        return None

def feature_level_fusion(features1, features2, target_dim=100):
    try:
        if features1 is None or features2 is None:
            print("No features have been provided for fusion.")
            return None

        # Truncate to the minimum number of samples
        min_samples = min(features1.shape[0], features2.shape[0])
        features1 = features1[:min_samples]
        features2 = features2[:min_samples]

        # Use PCA to reduce dimensionality to target_dim
        # First, handle features1
        if features1.shape[1] != target_dim:
            n_components = min(target_dim, features1.shape[1], min_samples)
            pca1 = PCA(n_components=n_components)
            features1_reduced = pca1.fit_transform(features1)
            # If PCA reduces to less than target_dim, pad with zeros (this should be rare)
            if features1_reduced.shape[1] < target_dim:
                features1_reduced = np.pad(
                    features1_reduced,
                    ((0, 0), (0, target_dim - features1_reduced.shape[1])),
                    mode='constant',
                    constant_values=0
                )
            features1 = features1_reduced

        # Handle features2
        if features2.shape[1] != target_dim:
            n_components = min(target_dim, features2.shape[1], min_samples)
            pca2 = PCA(n_components=n_components)
            features2_reduced = pca2.fit_transform(features2)
            if features2_reduced.shape[1] < target_dim:
                features2_reduced = np.pad(
                    features2_reduced,
                    ((0, 0), (0, target_dim - features2_reduced.shape[1])),
                    mode='constant',
                    constant_values=0
                )
            features2 = features2_reduced

        # Ensure shapes match target_dim
        features1 = features1[:, :target_dim]
        features2 = features2[:, :target_dim]

        # Calculate variance for each feature dimension
        var1 = np.var(features1, axis=0, ddof=1)
        var2 = np.var(features2, axis=0, ddof=1)

        # Avoid division by zero by setting a minimum variance threshold
        var1[var1 < 1e-10] = 1e-10
        var2[var2 < 1e-10] = 1e-10

        # Compute inverse-variance weights
        weight1 = 1 / var1
        weight2 = 1 / var2

        # Normalize weights so that they sum to 1
        total_weight = weight1 + weight2
        weight1 /= total_weight
        weight2 /= total_weight

        # Perform weighted fusion
        fused_features = weight1 * features1 + weight2 * features2

        return fused_features

    except Exception as e:
        print(f"Error in feature-level fusion: {e}")
        return None


def reduce_dimensionality(features, n_components=20):
    try:
        if features is None or features.shape[0] < 2 or features.shape[1] == 0:
            return None, None
        n_components = min(n_components, features.shape[0] - 1, features.shape[1])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(features), pca
    except Exception as e:
        print(f"Dimensionality reduction error: {e}")
        return None, None

def plot_metrics(train_acc, val_acc, output_dir):
    try:
        epochs = range(1, len(train_acc) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
        plt.plot(epochs, val_acc, 'r-', label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'accuracy_metrics.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting metrics: {e}")

def adjust_learning_rate(epoch, initial_lr=0.05, min_lr=0.001):
    lr = initial_lr * (0.5 ** (epoch // 5))  # Halve every 5 epochs
    return max(lr, min_lr)

def train_and_evaluate(folder1, folder2, output_dir, train_batches, val_batches, test_batches, epochs):
    os.makedirs(output_dir, exist_ok=True)
    
    all_labels = []
    train_features_list = []
    train_labels_list = []
    val_features_list = []
    val_labels_list = []
    test_features_list = []
    test_labels_list = []
    
    # Load training data
    for batch_idx in range(train_batches):
        features1, categories1 = load_batch_data_folder1(folder1, batch_idx, 'train')
        features2, labels2 = load_batch_data_folder2(folder2, batch_idx, 'train')
        if any(x is None for x in [features1, features2, categories1, labels2]):
            continue
        labels1 = np.argmax(categories1, axis=1) if categories1.ndim > 1 else np.round(categories1).astype(int)
        min_samples = min(len(labels1), len(labels2))
        all_labels.extend(labels1[:min_samples])
        fused_features = feature_level_fusion(normalize_features(features1), normalize_features(features2))
        if fused_features is not None and min_samples >= 2:
            train_features_list.append(fused_features[:min_samples])
            train_labels_list.extend(labels1[:min_samples])

    # Load validation data
    for batch_idx in range(val_batches):
        features1, categories1 = load_batch_data_folder1(folder1, batch_idx, 'val')
        features2, labels2 = load_batch_data_folder2(folder2, batch_idx, 'val')
        if any(x is None for x in [features1, features2, categories1, labels2]):
            continue
        labels1 = np.argmax(categories1, axis=1) if categories1.ndim > 1 else np.round(categories1).astype(int)
        min_samples = min(len(labels1), len(labels2))
        all_labels.extend(labels1[:min_samples])
        fused_features = feature_level_fusion(normalize_features(features1), normalize_features(features2))
        if fused_features is not None and min_samples >= 2:
            val_features_list.append(fused_features[:min_samples])
            val_labels_list.extend(labels1[:min_samples])

    # Load test data
    for batch_idx in range(test_batches):
        features1, categories1 = load_batch_data_folder1(folder1, batch_idx, 'test')
        features2, labels2 = load_batch_data_folder2(folder2, batch_idx, 'test')
        if any(x is None for x in [features1, features2, categories1, labels2]):
            continue
        labels1 = np.argmax(categories1, axis=1) if categories1.ndim > 1 else np.round(categories1).astype(int)
        min_samples = min(len(labels1), len(labels2))
        all_labels.extend(labels1[:min_samples])
        fused_features = feature_level_fusion(normalize_features(features1), normalize_features(features2))
        if fused_features is not None and min_samples >= 1:
            test_features_list.append(fused_features[:min_samples])
            test_labels_list.extend(labels1[:min_samples])

    if not all_labels:
        print("No valid labels found. Exiting.")
        return

    le = LabelEncoder()
    le.fit(all_labels)
    num_classes = len(le.classes_)
    print(f"Total classes: {num_classes}")
    logging.info(f"Total classes: {num_classes}")
    
    # Apply PCA for dimensionality reduction
    if train_features_list:
        all_train_features = np.vstack(train_features_list)
        n_samples, n_features = all_train_features.shape
        n_components = min(n_samples, n_features, 16)
        pca = PCA(n_components=n_components)
        pca.fit(all_train_features)
        
        reduced_train_features = [pca.transform(batch) for batch in train_features_list]
        reduced_val_features = [pca.transform(batch) for batch in val_features_list] if val_features_list else []
        reduced_test_features = [pca.transform(batch) for batch in test_features_list] if test_features_list else []
    else:
        print("No train features available. Exiting.")
        return
    
    train_labels_array = np.array(train_labels_list)
    val_labels_array = np.array(val_labels_list) if val_labels_list else np.array([])
    test_labels_array = np.array(test_labels_list) if test_labels_list else np.array([])
    
    train_labels_encoded = le.transform(train_labels_array)
    val_labels_encoded = le.transform(val_labels_array) if val_labels_array.size > 0 else np.array([])
    test_labels_encoded = le.transform(test_labels_array) if test_labels_array.size > 0 else np.array([])
    
    # Compute class weights for imbalanced classes
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(train_labels_encoded)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels_encoded)
    class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
    
    # Initialize base models
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        max_features='sqrt',
        n_jobs=-1,
        class_weight=class_weight_dict,
        warm_start=True
    )
    
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=31,
        objective='multiclass',
        num_class=num_classes,
        n_jobs=-1,
        verbose=-1,
        boosting_type='gbdt',
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0
    )
    
    # Initial fit on a small batch to warm-start
    if reduced_train_features:
        small_batch = reduced_train_features[0]
        small_labels = train_labels_encoded[:len(small_batch)]
        rf.fit(small_batch, small_labels)
        lgbm.fit(small_batch, small_labels)
    
    train_acc_history = []
    val_acc_history = []
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    current_lr = 0.05
    best_voting_clf = None
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Current Learning Rate: {current_lr:.4f}")
        
        # Update RandomForestClassifier n_estimators using warm_start
        rf.n_estimators += 50  # Incrementally increase trees
        
        # Update LGBMClassifier with current learning rate
        lgbm.set_params(learning_rate=current_lr)
        
        # Fit models on all training data
        all_train_features = np.vstack(reduced_train_features)
        all_train_labels = train_labels_encoded
        
        rf.fit(all_train_features, all_train_labels)
        lgbm.fit(all_train_features, all_train_labels)
        
        # Create VotingClassifier with fitted models
        voting_clf = VotingClassifier(
            estimators=[('rf', rf), ('lgbm', lgbm)],
            voting='soft',
            weights=[0.4, 0.6],
            n_jobs=-1
        )
        
        voting_clf.fit(all_train_features, all_train_labels)
        
        # Evaluate on training data
        train_preds = voting_clf.predict(all_train_features)
        train_acc = accuracy_score(all_train_labels, train_preds)
        print(f"Average Train - Accuracy: {train_acc:.4f}")
        logging.info(f"Epoch {epoch + 1} - Train Accuracy: {train_acc:.4f}")
        
        # Evaluate on validation data
        val_acc = 0.0
        if reduced_val_features and len(val_labels_encoded) > 0:
            all_val_features = np.vstack(reduced_val_features)
            val_preds = voting_clf.predict(all_val_features)
            val_acc = accuracy_score(val_labels_encoded, val_preds)
            print(f"Validation - Accuracy: {val_acc:.4f}")
            logging.info(f"Epoch {epoch + 1} - Validation Accuracy: {val_acc:.4f}")
        else:
            print("No validation data available")
        
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        # Learning rate scheduling
        if epoch > 0:
            if val_acc < val_acc_history[-2]:
                current_lr *= 0.7
            elif train_acc > 0.95:
                current_lr *= 0.8
            elif val_acc > val_acc_history[-2] and train_acc < 0.9:
                current_lr *= 1.1
        
        current_lr = max(0.001, min(0.1, current_lr))
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_voting_clf = voting_clf
            print("Updated best model based on validation accuracy")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
    
    # Use the best voting classifier for final evaluation
    if best_voting_clf is None:
        best_voting_clf = voting_clf
        print("No better model found; using the last voting classifier")
    
    # Save models
    try:
        with open(os.path.join(output_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(best_voting_clf.named_estimators_['rf'], f)
        with open(os.path.join(output_dir, 'lgbm_model.pkl'), 'wb') as f:
            pickle.dump(best_voting_clf.named_estimators_['lgbm'], f)
        with open(os.path.join(output_dir, 'voting_model.pkl'), 'wb') as f:
            pickle.dump(best_voting_clf, f)
        print(f"Models saved to {output_dir}")
        logging.info(f"Models saved to {output_dir}")
    except Exception as e:
        print(f"Error saving models: {e}")
        logging.error(f"Error saving models: {e}")
    
    # Evaluate on test data
    if reduced_test_features and len(test_labels_encoded) > 0:
        all_test_features = np.vstack(reduced_test_features)
        test_preds = best_voting_clf.predict(all_test_features)
        test_acc = accuracy_score(test_labels_encoded, test_preds)
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        logging.info(f"Final Test Accuracy: {test_acc:.4f}")
        
        try:
            print("\nClassification Report:")
            report = classification_report(
                test_labels_encoded,
                test_preds,
                labels=range(num_classes),
                target_names=[str(cls) for cls in le.classes_],  
                zero_division=0
            )
            print(report)
            logging.info(f"Classification Report:\n{report}")
            
            cm = confusion_matrix(test_labels_encoded, test_preds, labels=range(num_classes))
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xticks(range(num_classes), le.classes_, rotation=45)
            plt.yticks(range(num_classes), le.classes_)
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
        
        except Exception as e:
            print(f"Error in Test Evaluation: {e}")
            logging.error(f"Error in Test Evaluation: {e}")
    
    if train_acc_history:
        plot_metrics(train_acc_history, val_acc_history, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, default="fused_reduced_features")
    parser.add_argument("--folder2", type=str, default="fused_reduced_features")
    parser.add_argument("--output_dir", type=str, default="model_output")
    parser.add_argument("--train_batches", type=int, default=70)
    parser.add_argument("--val_batches", type=int, default=18)
    parser.add_argument("--test_batches", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=10)  # Increased to allow early stopping
    args = parser.parse_args()

    train_and_evaluate(
        args.folder1, args.folder2, args.output_dir,
        args.train_batches, args.val_batches, args.test_batches, args.epochs
    )