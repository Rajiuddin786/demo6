import pickle
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

from utils.feature_extractor import AdvancedFeatureExtractor

models=['models/randomForestModel.pkl','models/lightbgmModel.pkl','models/xgboostModel.pkl']
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
with open(models[2],'rb') as f:
    model=pickle.load(f)

scaler=StandardScaler()
df=pd.read_csv('data/sample20.csv')
features=AdvancedFeatureExtractor().extract_comprehensive_features(df)
feature_vector=pd.DataFrame([features])

feature_names=['dwell_mean','flight_mean','press_press_mean','release_release_mean',
               'total_time','dwell_std','flight_std','press_press_std','release_release_std','dwell_cv','flight_cv',
                'release_release_cv','press_press_cv','dwell_flight_ratio','pause_ratio']

for feature in feature_names:
    if feature not in feature_vector.columns:
        feature_vector[feature] = 0.0

feature_vector = feature_vector[feature_names]
feature_vector = feature_vector.fillna(0)

# Scale features
X_scaled = scaler.fit_transform(feature_vector)

# Get predictions from all models
predictions = {}
probabilities = {}
pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0]

percentage = proba[1] * 100
print(f"Confidence: {percentage:.2f}%")
