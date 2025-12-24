from data_tools import load_data, clean_data, engineer_features, transform_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_model():
    df = load_data()
    print("Starting model training with the Superstore dataset...")
    print("Dataset loaded:", len(df), "rows,", len(df.columns), "columns")
    print("Columns:", df.columns.tolist())

    required_columns = ['Sales', 'Quantity', 'Discount', 'Order Date']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for the Superstore model: {missing}")

    df = clean_data(df)
    df = engineer_features(df)

    X_scaled, y = transform_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sales_model.pkl")

    print("Model trained and saved successfully")
    return model, r2


if __name__ == "__main__":
    train_model()
