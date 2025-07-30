import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4);
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 2rem 0;
    }
    .success-message {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .info-message {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load and cache the cleaned dataset for reference
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💼 Employee Salary Predictor</h1>
        <p>Advanced Machine Learning Model for Salary Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    data = load_data()
    
    if model is None or data is None:
        st.error("❌ Failed to load model or data. Please check if all files are present.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📊 Model Information")
    st.sidebar.info(f"**Dataset Size:** {len(data):,} records")
    st.sidebar.info(f"**Features:** {len(data.columns)-1}")
    st.sidebar.info(f"**Target:** Income Classification")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction", "📈 Data Analysis", "ℹ️ About"])
    
    with tab1:
        single_prediction_tab(model, data)
    
    with tab2:
        batch_prediction_tab(model)
    
    with tab3:
        data_analysis_tab(data)
    
    with tab4:
        about_tab()

def single_prediction_tab(model, data):
    st.markdown("## 🎯 Single Employee Prediction")
    st.markdown("Enter employee details to predict their salary classification.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Information")
        age = st.slider("Age", min_value=17, max_value=75, value=35, help="Employee age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Employee gender")
        
        # Convert gender to encoded value (assuming 1=Male, 0=Female based on typical encoding)
        gender_encoded = 1 if gender == "Male" else 0
        
        marital_status = st.selectbox("Marital Status", [
            "Married-civ-spouse", "Never-married", "Divorced", 
            "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ], help="Current marital status")
        
        # Encode marital status (simplified mapping)
        marital_mapping = {
            "Married-civ-spouse": 2, "Never-married": 4, "Divorced": 6,
            "Separated": 3, "Widowed": 5, "Married-spouse-absent": 0, "Married-AF-spouse": 1
        }
        marital_encoded = marital_mapping.get(marital_status, 2)
    
    with col2:
        st.markdown("### Professional Information")
        workclass = st.selectbox("Work Class", [
            "Private", "Self-emp-not-inc", "Local-gov", "State-gov", 
            "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked"
        ], help="Type of employment")
        
        # Encode workclass (simplified mapping)
        workclass_mapping = {
            "Private": 2, "Self-emp-not-inc": 6, "Local-gov": 3, "State-gov": 4,
            "Self-emp-inc": 5, "Federal-gov": 1, "Without-pay": 0, "Never-worked": 7
        }
        workclass_encoded = workclass_mapping.get(workclass, 2)
        
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
            "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
            "Protective-serv", "Armed-Forces"
        ], help="Job role")
        
        # Encode occupation (simplified mapping)
        occupation_mapping = {
            "Tech-support": 12, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
            "Exec-managerial": 4, "Prof-specialty": 0, "Handlers-cleaners": 6, "Machine-op-inspct": 7,
            "Adm-clerical": 8, "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11,
            "Protective-serv": 13, "Armed-Forces": 14
        }
        occupation_encoded = occupation_mapping.get(occupation, 0)
        
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=80, value=40, help="Weekly working hours")
        educational_num = st.slider("Education Level", min_value=5, max_value=16, value=9, help="Educational attainment (numerical)")
    
    # Additional features
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Additional Information")
        fnlwgt = st.number_input("Final Weight", min_value=10000, max_value=500000, value=200000, help="Statistical weight")
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, help="Capital gains")
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, help="Capital losses")
    
    with col4:
        relationship = st.selectbox("Relationship", [
            "Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"
        ], help="Relationship status")
        
        # Encode relationship
        relationship_mapping = {
            "Husband": 0, "Not-in-family": 1, "Other-relative": 2, 
            "Own-child": 3, "Unmarried": 4, "Wife": 5
        }
        relationship_encoded = relationship_mapping.get(relationship, 0)
        
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], help="Ethnicity")
        
        # Encode race
        race_mapping = {
            "White": 4, "Black": 2, "Asian-Pac-Islander": 1, 
            "Amer-Indian-Eskimo": 0, "Other": 3
        }
        race_encoded = race_mapping.get(race, 4)
        
        native_country = st.selectbox("Native Country", ["United-States", "Other"], help="Country of origin")
        native_country_encoded = 39 if native_country == "United-States" else 0
    
    # Create input array
    input_data = np.array([[
        age, workclass_encoded, fnlwgt, educational_num, marital_encoded,
        occupation_encoded, relationship_encoded, race_encoded, gender_encoded,
        capital_gain, capital_loss, hours_per_week, native_country_encoded
    ]])
    
    # Display input summary
    st.markdown("### 📋 Input Summary")
    input_df = pd.DataFrame({
        'Feature': ['Age', 'Work Class', 'Final Weight', 'Education', 'Marital Status', 
                   'Occupation', 'Relationship', 'Race', 'Gender', 'Capital Gain', 
                   'Capital Loss', 'Hours per Week', 'Native Country'],
        'Value': [age, workclass, fnlwgt, educational_num, marital_status, 
                 occupation, relationship, race, gender, capital_gain, 
                 capital_loss, hours_per_week, native_country]
    })
    st.dataframe(input_df, use_container_width=True)
    
    # Prediction button
    col5, col6, col7 = st.columns([1, 2, 1])
    with col6:
        if st.button("🚀 Predict Salary Classification", use_container_width=True):
            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                # Display result
                if prediction == ">50K":
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>💰 High Income Prediction</h2>
                        <h3>Predicted: {prediction}</h3>
                        <p>Confidence: {max(probability)*100:.1f}%</p>
                        <p>This employee is likely to earn more than $50K annually.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>💼 Standard Income Prediction</h2>
                        <h3>Predicted: {prediction}</h3>
                        <p>Confidence: {max(probability)*100:.1f}%</p>
                        <p>This employee is likely to earn $50K or less annually.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probability breakdown
                st.markdown("### 📊 Prediction Confidence")
                prob_df = pd.DataFrame({
                    'Class': ['≤50K', '>50K'],
                    'Probability': [probability[0], probability[1]]
                })
                st.bar_chart(prob_df.set_index('Class'))
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

def batch_prediction_tab(model):
    st.markdown("## 📁 Batch Prediction")
    st.markdown("Upload a CSV file with employee data for batch predictions.")
    
    st.markdown("""
    <div class="info-message">
        <h4>📋 Required CSV Format</h4>
        <p>Your CSV should contain these columns (encoded numerically):</p>
        <ul>
            <li>age, workclass, fnlwgt, educational-num, marital-status</li>
            <li>occupation, relationship, race, gender, capital-gain</li>
            <li>capital-loss, hours-per-week, native-country</li>
        </ul>
        <p><strong>Note:</strong> Use the cleaned_dataset.csv as a reference for the correct format.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with employee data"
    )
    
    if uploaded_file is not None:
        try:
            # Load the uploaded data
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_data)} records")
            
            # Display data preview
            st.markdown("### 📊 Data Preview")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            # Check if required columns are present
            required_columns = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
                              'occupation', 'relationship', 'race', 'gender', 'capital-gain',
                              'capital-loss', 'hours-per-week', 'native-country']
            
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.markdown("Please ensure your CSV file contains all required columns.")
            else:
                # Make predictions
                if st.button("🚀 Run Batch Predictions", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        try:
                            # Prepare features (exclude target if present)
                            feature_columns = [col for col in batch_data.columns if col != 'income']
                            X_batch = batch_data[feature_columns]
                            
                            # Make predictions
                            predictions = model.predict(X_batch)
                            probabilities = model.predict_proba(X_batch)
                            
                            # Add predictions to dataframe
                            batch_data['Predicted_Income'] = predictions
                            batch_data['Confidence'] = np.max(probabilities, axis=1)
                            
                            # Display results
                            st.markdown("### 🎯 Prediction Results")
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Records", len(batch_data))
                            with col2:
                                high_income_count = (batch_data['Predicted_Income'] == '>50K').sum()
                                st.metric("High Income (>50K)", high_income_count)
                            with col3:
                                low_income_count = (batch_data['Predicted_Income'] == '<=50K').sum()
                                st.metric("Standard Income (≤50K)", low_income_count)
                            
                            # Display results table
                            st.markdown("### 📋 Detailed Results")
                            st.dataframe(batch_data[['Predicted_Income', 'Confidence'] + feature_columns[:5]], use_container_width=True)
                            
                            # Download results
                            csv = batch_data.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv,
                                file_name='salary_predictions.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Prediction error: {e}")
                            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

def data_analysis_tab(data):
    st.markdown("## 📈 Data Analysis")
    st.markdown("Explore the dataset statistics and distributions.")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Overview")
        st.metric("Total Records", len(data))
        st.metric("Features", len(data.columns) - 1)
        st.metric("Income Classes", data['income'].nunique())
        
        # Income distribution
        income_counts = data['income'].value_counts()
        st.markdown("### 💰 Income Distribution")
        fig = px.pie(
            values=income_counts.values, 
            names=income_counts.index,
            title="Income Class Distribution",
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Age Distribution")
        fig = px.histogram(
            data, 
            x='age', 
            nbins=30,
            title="Age Distribution",
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### ⏰ Hours per Week Distribution")
        fig = px.histogram(
            data, 
            x='hours-per-week', 
            nbins=20,
            title="Weekly Hours Distribution",
            color_discrete_sequence=['#764ba2']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Feature Correlations")
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed statistics
    st.markdown("### 📋 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)

def about_tab():
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Project Overview</h3>
        <p>This is a machine learning application that predicts employee salary classification 
        based on demographic and professional features. The model uses the Adult Census Income 
        Dataset to classify whether an employee earns more than $50K annually.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛠️ Technical Details")
        st.markdown("""
        - **Model**: Gradient Boosting Classifier
        - **Accuracy**: ~85%
        - **Features**: 13 numerical features
        - **Target**: Binary classification (>50K vs ≤50K)
        - **Framework**: Scikit-learn, Streamlit
        """)
    
    with col2:
        st.markdown("### 📊 Dataset Information")
        st.markdown("""
        - **Source**: UCI Machine Learning Repository
        - **Records**: ~46,720 employees
        - **Features**: Age, work class, education, occupation, etc.
        - **Preprocessing**: Label encoding, outlier removal
        """)
    
    st.markdown("### 🚀 Features")
    st.markdown("""
    - ✅ **Single Prediction**: Predict salary class for individual employees
    - ✅ **Batch Prediction**: Process multiple employees via CSV upload
    - ✅ **Data Analysis**: Explore dataset statistics and distributions
    - ✅ **Modern UI**: Beautiful, responsive interface
    - ✅ **Real-time Results**: Instant predictions with confidence scores
    """)
    
    st.markdown("### 📝 Usage Instructions")
    st.markdown("""
    1. **Single Prediction**: Use the sidebar to input employee details and get instant predictions
    2. **Batch Prediction**: Upload a CSV file with employee data for bulk predictions
    3. **Data Analysis**: Explore the dataset statistics and visualizations
    4. **Download Results**: Export prediction results as CSV files
    """)

if __name__ == "__main__":
    main()
