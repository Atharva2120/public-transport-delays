from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Keep only numeric columns
    X = X.select_dtypes(include=['int64', 'float64'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'outputs/model.pkl')

    return model, X_test, y_test