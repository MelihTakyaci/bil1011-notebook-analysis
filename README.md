# 📊 BIL-1011 Grade Prediction Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Data Mining Project for BIL-1011 Introduction to Computer Science I**

*Predicting midterm (vize) grades using regression models*

[🔗 Live Web App](https://github.com/MelihTakyaci/grade-prediction-app) | [📓 Notebook Analysis](#notebook-overview)

</div>

---

## 📖 Project Overview

This project applies **data mining** and **machine learning** techniques to predict student midterm grades for the **BIL-1011 Introduction to Computer Science I** course. Using real academic data, we implement and compare three different regression models to understand the relationships between study habits and academic performance.

### 🎯 Objectives

- **Predict** student midterm grades based on study patterns
- **Compare** different regression model performances
- **Analyze** feature importance and correlations
- **Visualize** data patterns and model predictions
- **Deploy** models as a web application

---

## 🔬 Models Implemented

### 1. **Linear Regression** 📈
- **Features:** Study Hours per Week
- **Purpose:** Simple relationship between study time and grades
- **Use Case:** Quick predictions based on study commitment

### 2. **Polynomial Regression** 📊
- **Features:** Course Attempts (with polynomial transformation)
- **Purpose:** Capture non-linear patterns in retake performance
- **Use Case:** Understanding diminishing returns on retakes

### 3. **Multiple Linear Regression** 🎲
- **Features:** Study Hours, Course Attempts, Class Year
- **Purpose:** Comprehensive prediction using all available features
- **Use Case:** Most accurate predictions combining multiple factors

---

## 📁 Project Structure

```
notebook_report/
├── bil1011PredictModel.ipynb    # Main analysis notebook
├── BBG1.xlsx                     # Dataset (student grades & features)
├── requirements.txt              # Python dependencies
├── VeriMadenciligi2.Odevv1.docx # Project documentation (Turkish)
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MelihTakyaci/bil1011-notebook-analysis.git
   cd bil1011-notebook-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook bil1011PredictModel.ipynb
   ```

---

## 📊 Notebook Overview

The `bil1011PredictModel.ipynb` notebook contains:

### 1. **Data Loading & Exploration**
- Import dataset from Excel
- Examine data structure and statistics
- Check for missing values and outliers

### 2. **Exploratory Data Analysis (EDA)**
- Statistical summaries
- Correlation analysis
- Data visualizations (histograms, scatter plots, heatmaps)

### 3. **Data Preprocessing**
- Feature selection
- Train-test split
- Data normalization/standardization

### 4. **Model Training**
- Linear Regression implementation
- Polynomial Regression with degree optimization
- Multiple Linear Regression

### 5. **Model Evaluation**
- Performance metrics (R², MSE, RMSE, MAE)
- Residual analysis
- Cross-validation results

### 6. **Visualization**
- Actual vs Predicted plots
- Feature importance charts
- Interactive Plotly visualizations

### 7. **Model Persistence**
- Save trained models using joblib
- Export for web application deployment

---

## 📦 Dependencies

```
pandas          # Data manipulation and analysis
numpy           # Numerical computing
matplotlib      # Static plotting
plotly          # Interactive visualizations
scikit-learn    # Machine learning library
statsmodels     # Statistical models
openpyxl        # Excel file handling
joblib          # Model serialization
```

Install all at once:
```bash
pip install pandas numpy matplotlib scikit-learn statsmodels plotly openpyxl joblib
```

---

## 🌐 Web Application

The trained models from this notebook are deployed in a **Next.js web application** for real-time predictions:

- **Repository:** [grade-prediction-app](https://github.com/MelihTakyaci/grade-prediction-app)
- **Features:**
  - Bilingual interface (English/Turkish)
  - Three regression models running simultaneously
  - Glassmorphism design with animations
  - Mobile-responsive with touch interactions
  - Data mining-inspired visual effects

---

## 📈 Results & Performance

### Model Comparison

| Model | Features | R² Score | Best Use Case |
|-------|----------|----------|---------------|
| **Linear Regression** | Study Hours | ~0.XX | Quick estimates |
| **Polynomial Regression** | Course Attempts | ~0.XX | Retake analysis |
| **Multiple Regression** | Study Hours + Attempts + Year | ~0.XX | Most accurate |

*Note: See notebook for actual performance metrics*

---

## 👨‍🏫 Course Information

**Course:** BIL-1011 Introduction to Computer Science I  
**Project:** Data Mining Introduction (Veri Madenciliğine Giriş)  
**Instructor:** Efendi Nasiboğlu  
**Institution:** Dokuz Eylül University

---

## 📝 Key Learnings

✅ **Data Preprocessing:** Handling real-world academic data  
✅ **Feature Engineering:** Selecting relevant predictors  
✅ **Model Selection:** Comparing regression techniques  
✅ **Evaluation Metrics:** Understanding R², MSE, RMSE, MAE  
✅ **Visualization:** Communicating results effectively  
✅ **Deployment:** Integrating ML models with web applications  

---

## 🔮 Future Improvements

- [ ] Add more features (attendance, assignment scores, etc.)
- [ ] Implement advanced models (Random Forest, XGBoost)
- [ ] Perform hyperparameter tuning
- [ ] Add cross-validation for all models
- [ ] Include confidence intervals for predictions
- [ ] Deploy notebook as interactive dashboard

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Course instructor **Efendi Nasiboğlu** for project guidance
- Dokuz Eylül University for providing the academic environment
- Open-source community for amazing ML libraries

---

## 📞 Contact

For questions or feedback, please reach out through GitHub issues or contact the team members.

---

<div align="center">

**Made with ❤️ for Data Mining Education**

⭐ Star this repo if you found it helpful!

</div>
