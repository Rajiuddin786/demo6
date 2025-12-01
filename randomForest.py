from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import Config
from utils.data_loader import DataLoader
from utils.feature_extractor import AdvancedFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomForest:
    def __init__(self):
        self.data_loader = DataLoader()
        self.scaler = StandardScaler()

    def train_model(self):
        X_train,y_train=self.prepare_data()

        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators= 300,
            max_depth =20,
            min_samples_split= 3,
            min_samples_leaf= 2,
            random_state= 42,
            class_weight='balanced')
        model.fit(X_train, y_train)
        
        model.fit(X_train, y_train)
        self.save_model(model)

    def prepare_data(self):
        # Load and prepare data
        X, y, file_info = self.data_loader.load_training_data()
        logger.info(f"Dataset shape: {X.shape}")
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled,y_train
    
    def handle_missing_values(self, X):
        """Handle missing values intelligently"""
        # For numeric columns, use median
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        return X
    
    def save_model(self,model):
        with open('models/randomForestModel.pkl','wb') as f:
            pickle.dump(model,f)

if __name__ == "__main__":
    RandomForest().train_model()