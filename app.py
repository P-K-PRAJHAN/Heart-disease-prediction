from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model and preprocessing objects
model = joblib.load('model/best_model.pkl')
scaler = joblib.load('model/scaler.pkl')
imputer = joblib.load('model/imputer.pkl')

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page"""
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    try:
        # Get input data from form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])
        
        # Create feature array
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Apply preprocessing
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Return result
        return render_template('result.html', 
                             prediction=prediction,
                             probability=round(probability * 100, 2))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Extract features
        age = float(data['age'])
        sex = float(data['sex'])
        cp = float(data['cp'])
        trestbps = float(data['trestbps'])
        chol = float(data['chol'])
        fbs = float(data['fbs'])
        restecg = float(data['restecg'])
        thalach = float(data['thalach'])
        exang = float(data['exang'])
        oldpeak = float(data['oldpeak'])
        slope = float(data['slope'])
        ca = float(data['ca'])
        thal = float(data['thal'])
        
        # Create feature array
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Apply preprocessing
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Return result
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk': 'High' if probability > 0.5 else 'Low'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)