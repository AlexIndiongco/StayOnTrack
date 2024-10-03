from flask import Flask, request, render_template, send_from_directory, session, request, redirect, url_for, flash
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(app.root_path, 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Secret key
app.secret_key = 'your_secret_key'

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

Session(app)

db = SQLAlchemy(app)

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Method to hash the password
def set_password(self, password):
    self.password = generate_password_hash(password)

# Method to check the password
def check_password(self, password):
    return check_password_hash(self.password, password)

# Create the database tables
with app.app_context():
    db.create_all()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

MODEL_FILE = 'models/pretrained_model_xgboost.pkl'
FEATURE_NAMES_FILE = 'models/feature_names.pkl'
IMPUTER_FILE = 'models/imputer.pkl'
TRAINING_GRAPHS_DIR = 'models/'

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
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        test_data = pd.read_excel(filepath)

        model = joblib.load(MODEL_FILE)
        feature_names = joblib.load(FEATURE_NAMES_FILE)
        imputer = joblib.load(IMPUTER_FILE)

        test_data['GPA'] = pd.to_numeric(test_data['GPA'], errors='coerce')

        test_data['GENDER'] = test_data['GENDER'].astype('category')
        test_data['ENROLLMENT STATUS'] = test_data['ENROLLMENT STATUS'].astype('category')
        test_data = pd.get_dummies(test_data, columns=['GENDER', 'ENROLLMENT STATUS'], drop_first=True)

        for col in feature_names:
            if col not in test_data.columns:
                test_data[col] = 0

        X_test = test_data[feature_names]

        X_test = imputer.transform(X_test)

        y_test_pred = model.predict(X_test)
        retention_prediction = y_test_pred[0] if len(y_test_pred) == 1 else None

        predicted_retained_students = sum(y_test_pred)
        total_students = len(y_test_pred)
        predicted_retention_rate = (predicted_retained_students / total_students) * 100

        if 'RETAINED' in test_data.columns:
            real_retained_students = sum(test_data['RETAINED'])
            real_retention_rate = (real_retained_students / len(test_data)) * 100
        else:
            real_retention_rate = None

        prediction_error = predicted_retention_rate - real_retention_rate if real_retention_rate is not None else None

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

        session['feature_importance_plot'] = feature_importance_plot
        session['correlation_heatmap_plot'] = correlation_heatmap_plot
        session['retention_by_course_plot'] = retention_by_course_plot
        session['accuracy_by_course_plot'] = accuracy_by_course_plot

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
    feature_importance_plot = session.get('feature_importance_plot')
    correlation_heatmap_plot = session.get('correlation_heatmap_plot')
    retention_by_course_plot = session.get('retention_by_course_plot')
    accuracy_by_course_plot = session.get('accuracy_by_course_plot')

    if not feature_importance_plot or not correlation_heatmap_plot or not retention_by_course_plot or not accuracy_by_course_plot:
        return "No plots available. Please upload a file first to generate the metrics."

    return render_template(
        'metricspage.html',
        feature_importance=feature_importance_plot,
        correlation_heatmap=correlation_heatmap_plot,
        retention_by_course=retention_by_course_plot,
        accuracy_by_course=accuracy_by_course_plot
    )

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the user exists in the database
        user = User.query.filter_by(username=username).first()
        
        # If user exists, check password
        if user and check_password_hash(user.password, password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('homepage'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        # Hash the password using pbkdf2:sha256
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()

        print(f"New user registered: {username}")  # Debugging line
        
        flash('User registered successfully!', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
