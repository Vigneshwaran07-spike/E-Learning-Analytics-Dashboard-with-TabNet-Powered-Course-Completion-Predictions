# 📊 E-Learning Analytics Dashboard with TabNet-Powered Course Completion Predictions

An interactive analytics dashboard for an e-learning platform designed to provide insights into course enrollments, user engagement, and completion rates. This project uses a **TabNet deep learning model** to predict the likelihood of course completion, helping enhance strategic decision-making for education platforms.

Developed as a showcase for the **Coursera Operations Associate** role, it highlights proficiency in AI, Data Engineering, Visualization, and Full-Stack Development.

---

## 🚀 Features

- **Popular Courses**: Bar chart of top-enrolled courses (>2 enrollments).
- **User Engagement Trends**: Monthly enrollment trends for 2025.
- **Course Completion Rates**: Pie chart of completion rates by category.
- **High-Performing Courses**: Lists courses with above-average ratings.
- **AI-Powered Predictions**: Predict course completion likelihood using TabNet model.

---

## 🧠 Tech Stack

- **Language**: Python 3.10
- **Machine Learning**: PyTorch, pytorch-tabnet, scikit-learn, imblearn (SMOTE)
- **Dashboard**: Streamlit
- **Database**: MySQL, SQLAlchemy, mysql-connector-python
- **Visualization**: Plotly
- **Others**: pandas, numpy, joblib

---

## 🗃️ Dataset Overview

Stored in MySQL (`elearning_platform`) with 500 records:
- **users**: User ID, name, join date
- **courses**: Course ID, name, rating, price, category
- **enrollments**: User ID, course ID, enrollment date, completion status

**Prediction Features**:
- Course rating, price, one-hot encoded category
- Days since user joined/enrolled
- User enrollment count, avg category rating, course popularity

---

## ⚙️ Installation

### ✅ Prerequisites
- Python 3.10
- MySQL Server
- Git

### 📥 Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/e-learning-platform-analytics.git
   cd e-learning-platform-analytics
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit==1.31.0 pandas==2.2.2 numpy sqlalchemy==2.0.23 \
               mysql-connector-python==7.0.2 plotly==3.10.0 joblib==1.3.1 \
               torch==2.2.2 pytorch-tabnet==4.1.0 scikit-learn==1.4.0 imblearn==0.12.3
   ```

4. **Configure the MySQL database:**
   - Create the database:
     ```sql
     CREATE DATABASE elearning_platform;
     ```
   - Import schema and data:
     ```bash
     mysql -u root -p elearning_platform < schema.sql
     ```

   - Update the MySQL credentials in `train_model.py` and `app.py`:
     ```python
     engine = create_engine("mysql+mysqlconnector://root:your_password@localhost/elearning_platform")
     ```

5. **Train the model:**
   ```bash
   python train_model.py
   ```
   - Outputs:
     - `best_model.zip` – trained TabNet model
     - `scaler.pkl` – preprocessor
     - `feature_columns.pkl` – selected features

6. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```
   - Open in browser: `http://localhost:8501`

---

## 🧭 Usage

- Navigate the sidebar to explore analytics (Popular Courses, Engagement, etc.)
- Go to **“Predict Completion Likelihood”**:
  - Select a user, course, and date
  - View predicted course completion probability (e.g., 78.5%)

---

## 🧾 Project Structure

```
.
├── app.py                  # Streamlit dashboard
├── train_model.py          # TabNet training script
├── best_model.zip          # Trained TabNet model
├── scaler.pkl              # Preprocessing scaler
├── feature_columns.pkl     # List of features used
├── schema.sql              # MySQL schema and sample data
└── README.md               # This file
```

---

## 📈 Performance

- **Model**: TabNet (accuracy ~73–80%)
- **Imbalance Handling**: SMOTE oversampling
- **Key Features**: Course popularity, category rating, enrollment count

---

## 🔮 Future Enhancements

- Add feature importance visualizations for TabNet
- Real-time data updates via API integration
- Experiment with alternative models (NODE, LightGBM)
- Personalized recommendations on the dashboard

---

## 📩 Contact

For questions or collaborations, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/vigneshwaran--r) or open an issue.

---

⭐ If you find this project useful, please consider giving it a star!
