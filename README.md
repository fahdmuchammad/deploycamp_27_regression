# 🚗 Capstone Project Documentation  

---

## 1. Project Overview  

### 📌 Project Title:  
**Car Price Regression**  

### 👥 Team Information:  
- Achmad Al Fauzi Dhiaulhaq  
- Arya Adikusuma  
- Muchammad Fahd Ishamuddin  

### 📑 Executive Summary  
This project develops a **predictive model for used car pricing in the U.S. market** using a dataset of **205 vehicles**. The main objective is to support **positioning strategies** for new entrants, such as Chinese automotive companies, seeking to compete effectively against established brands.  

The workflow covers **data cleaning, preprocessing, exploratory analysis, feature selection, and model evaluation**. After benchmarking several algorithms, the **Tuned XGBoost Regressor** provided the most reliable predictions, achieving competitive accuracy.  

This solution enables **dealers, manufacturers, and financial institutions** to leverage **data-driven insights** for pricing, ensuring more **transparent, rational, and competitive decisions** in the American automotive market.  

---

### 1.1 Problem & Solution  

**The Problem**  
An Auto Brand faces the challenge of identifying which vehicle, manufacturer, and market variables significantly predict car prices in the U.S. and quantifying how well those variables explain price variation. The output must help set **competitive locally manufactured MSRPs**.  

**Primary Users:** Geely executives, product/pricing teams, sales & marketing, operations, finance, dealers.  
**Secondary Users:** Regulators, consultants.  

Accurate, data-driven pricing affects:  
- Market entry viability  
- Margin and profitability  
- Product localization decisions  
- Competitive differentiation  
- Regulatory/incentive optimization  

**The Solution**  
A **machine learning pipeline** leveraging **XGBoost with Optuna tuning** was developed to generate **ranked predictors, interpretable outputs, and predictive recommendations**. Results are validated with **cross-validation and sensitivity analyses** to ensure **robust, actionable pricing insights**.  

---

### 1.2 Architecture  

#### System Overview  
The Car Price Predictor adopts a **modern microservices architecture** with full **MLOps integration**.  

- **Frontend:** Next.js (React, TypeScript, TailwindCSS)  
- **Backend:** FastAPI (Python, Pydantic validation)  
- **Database:** PostgreSQL (experiment metadata, user data)  
- **ML Framework:** MLflow, Scikit-learn, XGBoost  
- **Monitoring:** Prometheus + Grafana  
- **Storage:** MinIO (S3-compatible)  
- **Deployment:** Docker Compose  

#### Technology Stack  

| Layer          | Technology Used                                   | Justification                                                                 |
|----------------|---------------------------------------------------|-------------------------------------------------------------------------------|
| Front-End      | Next.js 15.4.6, React 19.1.0, TypeScript, Tailwind CSS | Modern, scalable UI framework with strong type safety                         |
| Back-End       | FastAPI, Python 3                                 | High-performance async API framework with auto-documentation                  |
| Database       | PostgreSQL 13                                     | Reliable relational DB with ACID compliance                                   |
| ML Framework   | MLflow, Scikit-learn, XGBoost                     | Experiment tracking, versioning, and ML lifecycle management                  |
| Monitoring     | Prometheus, Grafana                               | Real-time monitoring, alerting, and visualization                             |
| Storage        | MinIO (S3-compatible)                             | Secure artifact and dataset storage                                           |
| Deployment     | Docker Compose                                    | Consistent, portable container orchestration                                  |

---

## 2. Project Assets & Access  

- **Repository:** [GitHub](https://github.com/fahdmuchammad/deploycamp_27_regression.git)  
- **Video Demo:** [Google Drive](https://drive.google.com/file/d/1FdaJeRVy5I0bTR2PnMuKR3pkp2qnTuFL/view?usp=sharing)  
- **Live Application URLs:**  
  - http://103.47.224.217:3001/  
  - http://103.47.224.217:5000/  
  - http://103.47.224.217:3000/  
  - http://103.47.224.217:8080/docs  
  - http://103.47.224.217:9090/targets  
  - http://103.47.224.217:8001/browser/mlflow  

### Credentials  
- **Frontend:** `admin / admin123`  
- **MinIO:** `minio / minio123`  
- **Grafana:** `admin / muntilan`  
- **Postgres:** `mlflow / mlflow123`  

---

## 3. Detailed Technical Report  

### 3.1 Functionality  
- **Frontend App** for batch & single prediction.  
- **MLflow** for experiment tracking & deployment.  
- **Prometheus + Grafana** for monitoring.  
- **MinIO S3** for artifacts storage.  
- **FastAPI Backend** with endpoints:  
  - `/` : check model availability  
  - `/predict` : batch/single prediction  
  - `/refresh-model` : refresh deployed model  

### 3.2 Deployment  
- Docker Compose for orchestration  
- Ansible for CI/CD automation  

### 3.3 User Interaction  
- **API Documentation:** http://103.150.90.72:8080/docs#/  

### 3.4 Technical Implementation  
- **Algorithm:** XGBoost Regressor  
- **Dataset:** [Car Price Prediction – Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)  
- **Metrics:** R², MAE, MAPE, RMSE  
- **MLOps Practices:** MLflow tracking, Dockerized services, CI/CD pipeline  

---

## 4. Documentation & Verification  

| Test Case              | Steps                      | Expected Outcome                                  | Status |
|-------------------------|----------------------------|--------------------------------------------------|--------|
| API Documentation       | Verify endpoints           | Complete API docs with params & responses         | ✅ |
| Successful Login        | Navigate /login            | Redirect to dashboard                             | ✅ |
| Version Control         | Review Git strategy        | Clear branching & release documentation           | ✅ |
| ML Prediction           | Upload CSV & Analyze       | Prediction < 5s                                   | ✅ |
| Sequence Diagrams       | Review interaction flows   | User/API/ML pipeline clearly documented           | ✅ |
| Testing Documentation   | Review procedures          | Test cases & outcomes recorded                    | ✅ |
| Monitoring Setup        | Verify observability       | Prometheus + Grafana integrated                   | ✅ |
| Deployment              | Check infra setup          | Docker & environment variables documented         | ✅ |

---

## 5. Innovation & Impact  

### 🌟 Novelty  
Data-driven pricing strategy using **XGBoost + Optuna** on **205 U.S. car records**. Despite small data, provides actionable **pricing insights** for new automakers entering the U.S. market.  

### 🌍 Potential Market Impact  
- **Automakers:** Competitive, transparent market entry pricing  
- **Dealers/Platforms:** More accurate listings  
- **Consumers:** Objective price benchmarks  
- **Financial Institutions:** Reliable residual value for credit & insurance  

### ⚙️ Technical Challenges  
- Limited dataset size (205) → generalization issue  
- High cardinality categorical features  
- Extreme outliers impacting stability  
- Tuning risk of overfitting with small data  

### 🚀 Future Improvements  
- Expand dataset with real U.S. dealership & marketplace data  
- Add features: service history, vehicle condition, location, seasonality  
- Explore LightGBM, CatBoost, Deep Learning  
- Continuous retraining & feedback loops from real transactions  

---

## 6. Performance Metrics  
- **Accuracy Metrics:** R², MAE, MAPE, RMSE  
- **System Metrics:** Latency, throughput, resource usage  

---

## 7. Lessons Learned  

### ⚙️ Technical Lessons  
- Data quality & preprocessing crucial for stability  
- XGBoost + Optuna best for structured regression  
- MAPE essential for business interpretability  

### 🤝 Team Collaboration Insights  
- Clear role division boosted efficiency  
- Regular check-ins solved issues early  
- Knowledge sharing improved skills  
- Business alignment ensured relevance  

---

## 8. Appendices  

### 📂 Raw Data Sample  
```json
{ 
  "CarName": "alfa-romero giulia", 
  "fueltype": "gas", 
  "aspiration": "std", 
  "doornumber": "two", 
  "carbody": "convertible", 
  "drivewheel": "rwd", 
  "enginelocation": "front", 
  "wheelbase": 88.6, 
  "carheight": 48.8, 
  "enginetype": "dohc", 
  "cylindernumber": "four", 
  "fuelsystem": "mpfi", 
  "horsepower": 111, 
  "peakrpm": 5000, 
  "citympg": 21
}
