# 🕵️ Fake Job  Detector

A **Machine Learning + NLP powered web application** that detects whether a job posting is **Legitimate** or **Fraudulent**, along with **Explainable AI insights**.

Built using **Streamlit**, this project helps job seekers identify suspicious job listings based on textual patterns and risk signals.

---

# 🚀 Features

### 🔍 Core Functionality

* Detects **fraudulent job postings** in real-time
* Provides **fraud probability score**
* Classifies job as:

  * ✅ Legitimate
  * 🚨 Fraudulent

### 🧠 Explainable AI (XAI)

* Highlights **important words influencing prediction**
* Shows **top contributing features**
* Keyword-based risk analysis:

  * Fraud keywords
  * Trust keywords

### 📊 Visualization Dashboard

* Fraud risk gauge meter
* Confusion matrix
* ROC curve
* Dataset insights
* Feature importance graphs

### ⚡ Additional Features

* Risk signal breakdown
* Keyword detection system
* Interactive UI with Streamlit
* Model auto-training (if not preloaded)

---

# 🧠 How It Works

## 1. Input Fields

User provides:

* Job Title
* Company Profile
* Job Description
* Requirements
* Benefits
* Salary, Location, Employment Type

---

## 2. Text Preprocessing

* Lowercasing
* Removing URLs and special characters
* Cleaning whitespace
* Combining all fields into one text

---

## 3. Feature Engineering

* TF-IDF Vectorization

  * Max features: **15,000**
  * N-grams: **Unigrams + Bigrams**
  * Sublinear TF scaling

---

## 4. Model Training

* Algorithm: **Support Vector Machine**
* Class imbalance handled using:

  ```
  class_weight = "balanced"
  ```

---

## 5. Prediction

* Outputs:

  * Fraud Probability
  * Legitimate Probability
  * Final Classification

---

## 6. Explainable AI

* Uses:

  * TF-IDF scores
  * Model coefficients

* Computes:

  ```
  Importance = TF-IDF × Model Weights
  ```

* Highlights:

  * Top influential words
  * Fraud indicators in text

---

# 📊 Dataset

* **Name:** Fake Job Postings Dataset (EMSCAD)
* **Source:** Kaggle
* **Total Samples:** ~17,880
* **Fraudulent Jobs:** ~866 (~4.8%)
* **Legitimate Jobs:** ~17,014 (~95.2%)

---

# 🛠️ Tech Stack

## 🌐 Frontend

* Streamlit

## 📚 Libraries

* scikit-learn
* pandas
* numpy
* matplotlib
* plotly

## 🧠 NLP

* TF-IDF Vectorizer

---

# 📁 Project Structure

```
Fake-Job-Detector/
│
├── fake_job_detector_app.py      # Main Streamlit application
├── fake_job_postings.csv         # Dataset (required)
├── job_fraud_model.pkl           # Saved model (auto-generated)
├── README.md                     # Project documentation
```

---

# ▶️ Installation & Setup

## 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/fake-job-detector.git
cd fake-job-detector
```

---

## 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 3️⃣ Add Dataset

Download and place:

```
fake_job_postings.csv
```

in the project root folder.

---

## 4️⃣ Run the Application

```
streamlit run fake_job_detector_app.py
```

---

# 📊 Model Performance

* ✅ High Accuracy (~95%+)
* 📈 ROC-AUC Score: Strong
* ⚖️ Balanced classification

---

# ⚡ Risk Signals Used

## 🚨 Fraud Indicators

* "work from home"
* "easy money"
* "no experience"
* "guaranteed income"
* "investment required"
* "pay to apply"
* "urgent hiring"

## 🟢 Trust Indicators

* "bachelor degree"
* "years of experience"
* "health insurance"
* "professional development"
* "background check"

---

# 🧪 Test Cases

## ❌ Fraud Example

```
Title: Work From Home Data Entry
Salary: ₹50,000/week
Requirement: Pay ₹2000 registration fee
```

👉 Output: **Fraudulent**

---

## ✅ Legit Example

```
Title: Software Engineer
Company: Infosys
Requirement: 2+ years experience in Java
```

👉 Output: **Legitimate**

---

# 📊 Model Insights Page

Includes:

* Confusion Matrix
* ROC Curve
* Dataset Distribution
* Fraud by Employment Type
* Top Fraud Words
* Top Legit Words

---

# 💡 Key Learning Outcomes

* NLP-based classification
* Handling imbalanced datasets
* Feature engineering using TF-IDF
* Explainable AI techniques
* Building interactive ML apps with Streamlit

---

# 🚀 Future Improvements

* 🔥 Deep Learning (LSTM / BERT)
* 🌐 Real-time job scraping APIs
* 📊 Advanced SHAP explanations
* 🧠 Ensemble models
* 📱 Mobile-friendly UI

---

# ⚠️ Disclaimer

This tool provides **predictions based on patterns in historical data**.
It should be used as a **support system**, not a final authority.

---

# 👩‍💻 Author

**Anushka Chavan**
B.Tech Data Science (3rd Year)

---

# ⭐ Support

If you found this project useful:

* ⭐ Star the repo
* 🍴 Fork it
* 📢 Share it

---

# 📬 Contact

For queries or collaboration:

* GitHub: your-profile-link
* Email: [your-email@example.com](mailto:your-email@example.com)
