import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path



class DataPreprocessor:
    def __init__(self,img_size=(224,224)):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()

    def load_data_from_directory(self, directory_path):
        """"Load all images and label from directory"""

        images = []
        labels = []
        labels_name = []

        classes = sorted(os.listdir(directory_path))
        
        for class_name in classes:
            class_path = os.path.join(directory_path, class_name)
            if not os.path.isdir(class_path):
                continue

            labels_name.append(class_name)

            for img_file in os.listdir(class_path)[:10]:
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(class_path,img_file)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0

                        images.append(img_array)
                        labels.append(class_name)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

        encoded_labels = self.label_encoder.fit_transform(labels)
        return np.array(images), encoded_labels, labels_name

    def split_data(self, X, y , test_size=0.15, val_size=0.15, random_state=42):
        """Split data into training and testing sets"""
        #First Split into temp and test sets
        X_temp,X_test,y_temp,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state, stratify=y)

        val_size_adj = val_size / (1 - test_size)
        #Second Split temp into train and validation sets
        X_train,X_val,y_train,y_val = train_test_split(X_temp,y_temp,test_size=val_size_adj,random_state=random_state, stratify=y_temp)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_split_data(self, train_data, val_data, test_data, output_dir):
        """Save split data to disk"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        X_train,y_train = train_data
        X_val,y_val = val_data
        X_test,y_test = test_data

        np.savez(os.path.join(output_dir, 'train_data.npz'), X=X_train, y=y_train)
        np.savez(os.path.join(output_dir, 'val_data.npz'), X=X_val, y=y_val)
        np.savez(os.path.join(output_dir, 'test_data.npz'), X=X_test, y=y_test)

        #save label encoder classes
        import pickle
        with open(os.path.join(output_dir,'label_encoder.pkl'),'wb') as f:
            pickle.dump(self.label_encoder, f)


preprocessor = DataPreprocessor()

images, encoded_labels, labels_name = preprocessor.load_data_from_directory('data/raw')
train_data, val_data, test_data = preprocessor.split_data(images, encoded_labels)

preprocessor.save_split_data(train_data, val_data, test_data, 'data/processed')
