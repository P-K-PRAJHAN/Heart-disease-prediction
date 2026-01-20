"""
Heart Disease Prediction - Model Development Script
This script performs the same operations as the Jupyter notebook
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
import joblib
warnings.filterwarnings('ignore')

# Set plotting style
#plt.style.use('seaborn-v0_8')
#sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the dataset"""
    print("1. Loading and exploring data...")
    df = pd.read_csv('data/heart_disease.csv')
    
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())
    
    return df

def perform_eda(df):
    """Perform exploratory data analysis"""
    print("\n2. Performing exploratory data analysis...")
    
    # Convert target to binary (0 = No disease, 1 = Disease)
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    
    print("Target distribution after conversion:")
    print(df['target'].value_counts())
    
    return df

def preprocess_data(df):
    """Preprocess the data"""
    print("\n3. Preprocessing data...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training set shape:", X_train_scaled.shape)
    print("Test set shape:", X_test_scaled.shape)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer, X.columns

def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train and evaluate models"""
    print("\n4. Training and evaluating models...")
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print()
    
    return results

def save_best_model(results):
    """Save the best performing model"""
    print("\n5. Saving the best model...")
    
    # Select the best model based on ROC-AUC score
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"Best Model: {best_model_name}")
    print(f"ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")
    
    # Save the best model and scaler
    joblib.dump(best_model, 'model/best_model.pkl')
    joblib.dump(results[best_model_name]['model'], 'model/' + best_model_name.lower().replace(' ', '_') + '.pkl')
    
    print("\nModel saved successfully!")
    
    return best_model_name, best_model

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, model_path='model/best_model.pkl'):
    """
    Predict heart disease risk for a patient based on input features.
    """
    # Load the trained model and preprocessing objects
    model = joblib.load(model_path)
    
    # Create feature array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability

def main():
    """Main function to run the complete pipeline"""
    # Load and explore data
    df = load_and_explore_data()
    
    # Perform EDA
    df = perform_eda(df)
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer, feature_columns = preprocess_data(df)
    
    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save best model
    best_model_name, best_model = save_best_model(results)
    
    # Save preprocessing objects
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(imputer, 'model/imputer.pkl')
    
    # Test prediction function
    print("\n6. Testing prediction function...")
    prediction, probability = predict_heart_disease(
        age=63, sex=1, cp=1, trestbps=145, chol=233, fbs=1, restecg=2, 
        thalach=150, exang=0, oldpeak=2.3, slope=3, ca=0, thal=6
    )
    
    print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
    print(f"Probability of Heart Disease: {probability:.4f}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
