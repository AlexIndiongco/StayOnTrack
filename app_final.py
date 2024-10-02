from flask import Flask, request, render_template, send_from_directory, session
from flask_session import Session
import pandas as pd
import numpy as np
import joblib
import os
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'  

# Set up server-side session using Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data on the server
app.config['SESSION_FILE_DIR'] = os.path.join(app.root_path, 'flask_session')  # Directory for session files
app.config['SESSION_PERMANENT'] = False  # Make sessions temporary
app.config['SESSION_USE_SIGNER'] = True  # Sign the session ID to prevent tampering
Session(app)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Paths for model and metadata files
MODEL_FILE = 'models/pretrained_model_xgboost.pkl'
FEATURE_NAMES_FILE = 'models/feature_names.pkl'
IMPUTER_FILE = 'models/imputer.pkl'
TRAINING_GRAPHS_DIR = 'models/'  # Directory where pre-trained graphs are saved

# Define the color mapping for each course
course_colors = {
    "Architecture": "#FF6F61",
    "BSAeronautical": "#6B5B95",
    "CE": "#88B04B",
    "Comp. Eng'g.": "#FFA500",
    "EE": "#92A8D1",
    "ELECENG": "#034F84",
    "IE": "#F7786B",
    "ME": "#C94C4C"
}

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory('downloads', filename)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load the test dataset
        test_data = pd.read_excel(filepath)

        # Load the trained model and necessary components
        model = joblib.load(MODEL_FILE)
        feature_names = joblib.load(FEATURE_NAMES_FILE)
        imputer = joblib.load(IMPUTER_FILE)

        # Preprocess test data
        test_data['GPA'] = pd.to_numeric(test_data['GPA'], errors='coerce')

        # One-hot encode categorical variables (Include GENDER and ENROLLMENT STATUS)
        test_data['GENDER'] = test_data['GENDER'].astype('category')
        test_data['ENROLLMENT STATUS'] = test_data['ENROLLMENT STATUS'].astype('category')
        test_data = pd.get_dummies(test_data, columns=['GENDER', 'ENROLLMENT STATUS'], drop_first=True)

        # Ensure all one-hot encoded columns are present, even if not in the test data
        for col in feature_names:
            if col not in test_data.columns:
                test_data[col] = 0  # Add missing columns with a default value of 0

        # Reorder columns to match the training data feature order
        X_test = test_data[feature_names]

        # Apply the imputer (same as in training)
        X_test = imputer.transform(X_test)

        # Make predictions on the test data
        y_test_pred = model.predict(X_test)
        retention_prediction = y_test_pred[0] if len(y_test_pred) == 1 else None

        # Calculate predicted overall retention rate (based on model predictions)
        predicted_retained_students = sum(y_test_pred)  # Predicted retained students
        total_students = len(y_test_pred)
        predicted_retention_rate = (predicted_retained_students / total_students) * 100

        # Calculate the real overall retention rate (if available)
        if 'RETAINED' in test_data.columns:
            real_retained_students = sum(test_data['RETAINED'])  # Real retained students from the actual data
            real_retention_rate = (real_retained_students / len(test_data)) * 100
        else:
            real_retention_rate = None

        # Calculate the prediction error
        prediction_error = predicted_retention_rate - real_retention_rate if real_retention_rate is not None else None

        # Retention prediction summary by course with distribution calculation
        retention_by_course_summary = []
        total_predicted_retained = sum(y_test_pred)
        if 'COURSE' in test_data.columns:
            courses = test_data['COURSE'].unique()

            for course in courses:
                course_data = test_data[test_data['COURSE'] == course]
                X_course = imputer.transform(course_data[feature_names])
                y_pred_course = model.predict(X_course)
                retained_count = sum(y_pred_course)
                real_retained_count = sum(course_data['RETAINED']) if 'RETAINED' in course_data.columns else None
                retention_distribution = (retained_count / total_predicted_retained) * 100 if total_predicted_retained > 0 else None

                retention_by_course_summary.append({
                    'course': course,
                    'predicted_retained_students': retained_count,
                    'total_students': len(course_data),
                    'real_retained_students': real_retained_count,
                    'retention_distribution': retention_distribution
                })

        # Generate plots and save to session
        # Feature Importance Plot
        plt.figure(figsize=(8, 8))
        feature_importance = model.feature_importances_
        plt.pie(
            feature_importance,
            labels=feature_names,
            autopct='%1.1f%%',
            startangle=90,
            counterclock=False,
            wedgeprops={'edgecolor': 'black'},
            colors=['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#034F84', '#F7786B', '#C94C4C']
        )
        img_feature_importance = io.BytesIO()
        plt.savefig(img_feature_importance, format='png')
        img_feature_importance.seek(0)
        feature_importance_plot = base64.b64encode(img_feature_importance.getvalue()).decode('utf8')
        plt.close()

        # Correlation Heatmap
        numeric_data = test_data.select_dtypes(include=[np.float64, np.int64, np.uint8])
        numeric_data = pd.concat([numeric_data, test_data[[col for col in test_data.columns if 'GENDER' in col or 'ENROLLMENT STATUS' in col]]], axis=1)
        numeric_data = numeric_data.drop(columns=['YEAR LEVEL', 'STUDENT NO'], errors='ignore')
        corr_matrix = numeric_data.corr()
        plt.figure(figsize=(6, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        img_corr_heatmap = io.BytesIO()
        plt.savefig(img_corr_heatmap, format='png')
        img_corr_heatmap.seek(0)
        correlation_heatmap_plot = base64.b64encode(img_corr_heatmap.getvalue()).decode('utf8')
        plt.close()

        # Retention by Course Plot
        if 'COURSE' in test_data.columns:
            courses = test_data['COURSE'].unique()
            course_labels = []
            retention_rates = []
            for course in courses:
                course_data = test_data[test_data['COURSE'] == course]
                if len(course_data) < 5:
                    continue
                retention_rate = course_data['RETAINED'].mean() * 100
                course_labels.append(course)
                retention_rates.append(retention_rate)

            plt.figure(figsize=(8, 6))
            bars = plt.bar(course_labels, retention_rates, color=[course_colors.get(course, '#333333') for course in course_labels])
            for bar, rate, label in zip(bars, retention_rates, course_labels):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'{rate:.1f}%', ha='center', va='bottom', fontsize=14)
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, label, ha='center', va='center', rotation='vertical', fontsize=16, color='white')
            img_retention_rate = io.BytesIO()
            plt.savefig(img_retention_rate, format='png')
            img_retention_rate.seek(0)
            retention_by_course_plot = base64.b64encode(img_retention_rate.getvalue()).decode('utf8')
            plt.close()

        # Accuracy by Course Plot
        retained_accuracy = []
        course_labels_accuracy = []
        if 'COURSE' in test_data.columns:
            for course in courses:
                course_data = test_data[test_data['COURSE'] == course]
                if len(course_data) < 5:
                    continue
                X_course = imputer.transform(course_data[feature_names])
                y_course = course_data['RETAINED']
                y_pred_course = model.predict(X_course)
                accuracy_retained = accuracy_score(y_course, y_pred_course)
                retained_accuracy.append(accuracy_retained * 100)
                course_labels_accuracy.append(course)

            plt.figure(figsize=(8, 6))
            bars = plt.bar(course_labels_accuracy, retained_accuracy, color=[course_colors.get(course, '#333333') for course in course_labels_accuracy])
            for bar, accuracy, label in zip(bars, retained_accuracy, course_labels_accuracy):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'{accuracy:.1f}%', ha='center', va='bottom', fontsize=14)
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, label, ha='center', va='center', rotation='vertical', fontsize=16, color='white')
            img_accuracy_rate = io.BytesIO()
            plt.savefig(img_accuracy_rate, format='png')
            img_accuracy_rate.seek(0)
            accuracy_by_course_plot = base64.b64encode(img_accuracy_rate.getvalue()).decode('utf8')
            plt.close()

        # Store plots in session (server-side)
        session['feature_importance_plot'] = feature_importance_plot
        session['correlation_heatmap_plot'] = correlation_heatmap_plot
        session['retention_by_course_plot'] = retention_by_course_plot
        session['accuracy_by_course_plot'] = accuracy_by_course_plot

        # Render the output template
        return render_template(
            'output.html',
            retention_by_course_summary=retention_by_course_summary,
            retention_prediction=retention_prediction,
            feature_importance=feature_importance_plot,
            correlation_heatmap=correlation_heatmap_plot,
            retention_by_course=retention_by_course_plot,
            accuracy_by_course=accuracy_by_course_plot,
            predicted_retention_rate=predicted_retention_rate,
            real_retention_rate=real_retention_rate,
            prediction_error=prediction_error
        )

    return render_template('index.html')

@app.route('/metricspage', methods=['GET'])
def metricspage():
    # Retrieve plots from the session (server-side)
    feature_importance_plot = session.get('feature_importance_plot')
    correlation_heatmap_plot = session.get('correlation_heatmap_plot')
    retention_by_course_plot = session.get('retention_by_course_plot')
    accuracy_by_course_plot = session.get('accuracy_by_course_plot')

    # If any plots are missing, show a message
    if not feature_importance_plot or not correlation_heatmap_plot or not retention_by_course_plot or not accuracy_by_course_plot:
        return "No plots available. Please upload a file first to generate the metrics."

    # Render the metricspage.html with the available plots
    return render_template(
        'metricspage.html',
        feature_importance=feature_importance_plot,
        correlation_heatmap=correlation_heatmap_plot,
        retention_by_course=retention_by_course_plot,
        accuracy_by_course=accuracy_by_course_plot
    )


if __name__ == '__main__':
    app.run(debug=True)
