# 💼 Employee Salary Predictor

A sophisticated machine learning web application that predicts employee salary classification using advanced algorithms and a modern, high-fidelity user interface.

## 🎯 Project Overview

This application uses the **Adult Census Income Dataset** to predict whether an employee earns **more than $50K** or **$50K or less** annually. The model achieves approximately **85% accuracy** using a Gradient Boosting Classifier.

## ✨ Key Features

### 🎯 Single Prediction
- **Interactive Input Forms**: User-friendly sliders and dropdowns for employee details
- **Real-time Predictions**: Instant results with confidence scores
- **Visual Feedback**: Beautiful cards and charts for prediction results
- **Comprehensive Input**: All 13 features supported with proper encoding

### 📁 Batch Prediction
- **CSV Upload**: Process multiple employees at once
- **Format Validation**: Automatic checking of required columns
- **Detailed Results**: Complete prediction breakdown with confidence scores
- **Export Functionality**: Download results as CSV files

### 📈 Data Analysis
- **Interactive Visualizations**: Plotly charts for data exploration
- **Statistical Summary**: Comprehensive dataset statistics
- **Correlation Analysis**: Feature relationship insights
- **Distribution Plots**: Age, hours, and income distributions

### 🎨 Modern UI/UX
- **Responsive Design**: Works perfectly on all devices
- **Gradient Themes**: Beautiful color schemes and animations
- **Professional Layout**: Clean, organized interface
- **Intuitive Navigation**: Tab-based organization

## 🛠️ Technical Stack

- **Frontend**: Streamlit with custom CSS
- **Backend**: Python 3.8+
- **Machine Learning**: Scikit-learn, Gradient Boosting
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Model Persistence**: Joblib

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd employee-salary-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the application**
Open your browser and navigate to `http://localhost:8501`

## 📊 Dataset Information

### Features Used
- **Personal**: Age, Gender, Marital Status, Relationship, Race
- **Professional**: Work Class, Occupation, Hours per Week
- **Educational**: Education Level (numerical)
- **Financial**: Capital Gain, Capital Loss, Final Weight
- **Geographic**: Native Country

### Data Preprocessing
- **Label Encoding**: Categorical variables converted to numerical
- **Outlier Removal**: Age and education level filtering
- **Missing Value Handling**: Proper imputation strategies
- **Feature Scaling**: StandardScaler for optimal model performance

## 🧠 Model Performance

### Algorithm: Gradient Boosting Classifier
- **Accuracy**: ~85%
- **Precision**: High for both classes
- **Recall**: Balanced performance
- **F1-Score**: Optimized for business needs

### Model Features
- **Ensemble Learning**: Combines multiple weak learners
- **Feature Importance**: Automatic feature selection
- **Regularization**: Prevents overfitting
- **Cross-validation**: Robust performance estimation

## 📁 Project Structure

```
employee-salary-prediction/
├── app.py                          # Main Streamlit application
├── best_model.pkl                  # Trained ML model
├── cleaned_dataset.csv             # Preprocessed dataset
├── dataset.csv                     # Original dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── EMPLOYEE SALARY PREDICTION USING MACHINE LEARNING & DEEP LEARNING ALGORITHMS.ipynb  # Training notebook
```

## 🎨 UI/UX Improvements

### Modern Design Elements
- **Gradient Backgrounds**: Beautiful color transitions
- **Card-based Layout**: Clean, organized information display
- **Interactive Elements**: Hover effects and animations
- **Responsive Design**: Works on desktop, tablet, and mobile

### User Experience
- **Intuitive Navigation**: Tab-based organization
- **Real-time Feedback**: Instant predictions and visualizations
- **Error Handling**: Graceful error messages and validation
- **Loading States**: Progress indicators for long operations

## 🔧 Usage Instructions

### Single Prediction
1. Navigate to the "Single Prediction" tab
2. Fill in employee details using the sidebar controls
3. Click "Predict Salary Classification"
4. View results with confidence scores and visualizations

### Batch Prediction
1. Navigate to the "Batch Prediction" tab
2. Upload a CSV file with employee data
3. Ensure the file has the correct column format
4. Click "Run Batch Predictions"
5. Download results as CSV

### Data Analysis
1. Navigate to the "Data Analysis" tab
2. Explore dataset statistics and distributions
3. View correlation matrices and visualizations
4. Understand the data patterns and relationships

## 📈 Performance Metrics

### Model Accuracy
- **Overall Accuracy**: 85.73%
- **High Income (>50K)**: 74% precision, 62% recall
- **Standard Income (≤50K)**: 88% precision, 93% recall

### Business Impact
- **Reliable Predictions**: High confidence in salary classifications
- **Scalable Processing**: Handles both single and batch predictions
- **User-friendly Interface**: Accessible to non-technical users
- **Export Capabilities**: Easy integration with existing workflows

## 🔮 Future Enhancements

### Planned Improvements
- **Cloud Deployment**: AWS, Google Cloud, or Azure integration
- **API Development**: RESTful API for external integrations
- **Advanced Analytics**: More detailed performance dashboards
- **Model Versioning**: A/B testing for model improvements
- **Real-time Updates**: Live model retraining capabilities

### Feature Additions
- **Multi-language Support**: Internationalization
- **Advanced Filtering**: More sophisticated data exploration
- **Custom Visualizations**: User-defined chart types
- **Export Formats**: PDF, Excel, and other formats
- **User Authentication**: Multi-user support with roles

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Harshavardhan1636**  
BTech CSE (AI & ML)

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the dataset
- **Streamlit** for the web framework
- **Scikit-learn** for machine learning tools
- **Plotly** for interactive visualizations

---

⭐ **Star this repository if you find it helpful!**
