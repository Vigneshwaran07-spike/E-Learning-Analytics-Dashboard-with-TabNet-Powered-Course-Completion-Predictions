import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTE
import joblib
import torch

# Fetch data from MySQL
def fetch_data():
    try:
        engine = create_engine("mysql+mysqlconnector://root:root@localhost/elearning_platform")
        query = """
        SELECT 
            e.user_id,
            e.course_id,
            e.completion_status,
            c.rating,
            c.price,
            c.category,
            DATEDIFF(e.enrollment_date, u.join_date) AS days_since_joined,
            DATEDIFF('2025-05-28', e.enrollment_date) AS days_since_enrolled,
            (SELECT COUNT(*) FROM enrollments e2 WHERE e2.user_id = e.user_id) AS user_enrollment_count,
            (SELECT AVG(c2.rating) FROM courses c2 WHERE c2.category = c.category) AS avg_category_rating,
            (SELECT COUNT(*) FROM enrollments e3 WHERE e3.course_id = e.course_id) AS course_popularity
        FROM enrollments e
        JOIN courses c ON e.course_id = c.course_id
        JOIN users u ON e.user_id = u.user_id;
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as err:
        print(f"Database error: {err}")
        return None

# Main function
def main():
    # Fetch data
    df = fetch_data()
    if df is None:
        print("Failed to fetch data. Exiting.")
        return
    
    # Display class distribution
    print("Class Distribution:\n", df['completion_status'].value_counts(normalize=True))
    
    # Check for nulls
    if df.isnull().any().any():
        print("Null values found:\n", df.isnull().sum())
        df = df.fillna(0)  # Replace nulls with 0
    
    # Convert completion_status to binary (1 for Completed, 0 for others)
    df['completion_status'] = df['completion_status'].apply(lambda x: 1 if x == 'Completed' else 0)
    
    # One-hot encode category
    df = pd.get_dummies(df, columns=['category'], prefix='category', dummy_na=False)
    
    # Ensure one-hot columns are numeric
    category_columns = [col for col in df.columns if col.startswith('category_')]
    for col in category_columns:
        df[col] = df[col].astype(int)
    
    # Features and target
    feature_columns = ['rating', 'price'] + category_columns + [
        'days_since_joined', 'days_since_enrolled', 'user_enrollment_count',
        'avg_category_rating', 'course_popularity'
    ]
    X = df[feature_columns].values.astype(np.float64)  # Ensure float64
    y = df['completion_status'].values
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_indices = [0, 1] + [i for i, col in enumerate(feature_columns) if col in [
        'days_since_joined', 'days_since_enrolled', 'user_enrollment_count',
        'avg_category_rating', 'course_popularity'
    ]]
    X[:, numerical_indices] = scaler.fit_transform(X[:, numerical_indices])
    
    # Save scaler and feature columns
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Train TabNet model
    model = TabNetClassifier(
        n_d=8, n_a=8, n_steps=3, gamma=1.5, lambda_sparse=0.01,
        optimizer_params=dict(lr=0.01),
        scheduler_params=dict(step_size=10, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=10,
        seed=42
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=['test'],
        max_epochs=100,
        batch_size=32,
        virtual_batch_size=32,
        patience=10,
        num_workers=0,
        weights=1,
        drop_last=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:\n", feature_importance)
    
    # Save model
    model.save_model('best_model')
    print("Model saved as 'best_model.zip'")

if __name__ == "__main__":
    main()