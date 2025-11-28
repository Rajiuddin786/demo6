import logging
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd

from config import Config
from utils.data_loader import DataLoader
from utils.feature_extractor import AdvancedFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightBGMModel:
    def __init__(self):
        self.data_loader = DataLoader()
        self.scaler = StandardScaler()

    def train_model(self):
        X_train,y_train=self.prepare_data()

        logger.info("Training LightGBM...")
        model = LGBMClassifier(
            n_estimators =  500,
            max_depth= 12,
            learning_rate=0.05,
            random_state= 42,
            class_weight="balanced")
        
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
        with open('models/lighbgmModel.pkl','wb') as f:
            pickle.dump(model,f)

if __name__ == '__main__':
    lightBGM=LightBGMModel()
    lightBGM.train_model()