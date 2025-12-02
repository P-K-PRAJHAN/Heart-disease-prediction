# Heart Disease Prediction ML System

A machine learning web application for predicting heart disease risk based on medical parameters using Random Forest, SVM, and Logistic Regression models.

## 📋 Project Overview

This project implements a complete machine learning pipeline for heart disease prediction:
- **Data Analysis**: Exploratory data analysis on the UCI Heart Disease dataset
- **Model Training**: Comparison of Random Forest, SVM, and Logistic Regression models
- **Web Interface**: Flask-based web application for user interaction
- **API**: RESTful API for programmatic access
- **Visualization**: Model performance dashboard with charts and metrics

## 🏗️ Architecture

```
Dataset → EDA → Preprocessing → Train/Test → Model Comparison → Flask API → Web UI
```

## 🧰 Tech Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, Bootstrap, JavaScript

## 📁 Project Structure

```
heart-disease-ml/
│
├── data/
│   └── heart_disease.csv          # Dataset
│
├── model/
│   ├── best_model.pkl             # Best performing model
│   ├── scaler.pkl                 # Feature scaler
│   ├── imputer.pkl                # Missing value imputer
│   └── logistic_regression.pkl    # Logistic Regression model
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom styles
│   ├── js/
│   │   └── script.js              # Client-side scripts
│   └── *.png                      # Visualization images
│
├── templates/
│   ├── base.html                  # Base template
│   ├── index.html                 # Main prediction form
│   ├── result.html                # Prediction results
│   └── dashboard.html             # Model comparison dashboard
│
├── app.py                         # Flask web application
├── model_development.py           # Model training pipeline
├── download_dataset.py            # Dataset downloader
├── generate_visualizations.py     # Dashboard visualizations
├── notebook.ipynb                 # Jupyter notebook for analysis
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd heart-disease-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```bash
   python download_dataset.py
   ```

4. Train the models:
   ```bash
   python model_development.py
   ```

5. Generate visualizations:
   ```bash
   python generate_visualizations.py
   ```

### Running the Application

Start the Flask server:
```bash
python app.py
```

Visit `http://localhost:5000` in your browser to access the application.

## 🎯 Features

### 1. Heart Disease Prediction
- User-friendly form for inputting medical parameters
- Real-time risk prediction with probability scores
- Personalized health recommendations

### 2. Model Comparison Dashboard
- Performance metrics comparison (Accuracy, ROC-AUC)
- ROC curves visualization
- Feature importance analysis
- Confusion matrices

### 3. RESTful API
- Programmatic access to prediction functionality
- JSON-based request/response format

API endpoint: `POST /api/predict`

Example request:
```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 85.25% | 0.9491 |
| Random Forest | 86.89% | 0.9405 |
| SVM | 83.61% | 0.9394 |

## 📈 Key Features

The most important features for heart disease prediction are:
1. Number of major vessels (ca)
2. Thalassemia (thal)
3. Chest pain type (cp)
4. ST depression (oldpeak)
5. Maximum heart rate (thalach)

## ⚠️ Disclaimer

This prediction tool is based on a machine learning model and should not be considered as a definitive medical diagnosis. Always consult with a qualified healthcare professional for accurate diagnosis and treatment recommendations.

## 📄 Resume Bullet Points

- Built a Heart Disease Risk Prediction ML System using Random Forest and SVM with 85% accuracy, deployed via Flask web interface
- Performed EDA, feature engineering, model selection, ROC-AUC evaluation and confusion matrix visualization
- Developed a responsive web UI with Bootstrap and implemented RESTful API for model serving

## 📝 License

This project is for educational purposes only.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.