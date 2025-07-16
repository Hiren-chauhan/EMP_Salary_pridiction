import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
class SalaryPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoders = {}  # Stores label encoders for categorical features
        self.X_columns = None  # Stores the order of features used during training
        self.accuracy = 0  # Stores the model's accuracy
    def fit(self, df):
        """Trains the model on the given dataframe."""
        df = df.copy()  # Create a copy to avoid modifying the original DataFrame
        # Replace '?' with the mode of the column
        for col in df.columns:
            if df[col].dtype == "object":
                mode = df[col].mode()[0]  # Find the most frequent value
                df[col].replace("?", mode, inplace=True)  # Replace '?' with the mode
        # The 'fnlwgt' column is a weight and not useful for individual prediction.
        if "fnlwgt" in df.columns:
            df = df.drop("fnlwgt", axis=1)  # Remove the 'fnlwgt' column
        # Define categorical columns including the target variable 'income'
        categorical_cols = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "native-country",
            "income",
        ]
        for col in categorical_cols:  # Iterate through categorical columns
            if col in df.columns:
                le = LabelEncoder()  # Initialize a LabelEncoder
                df[col] = le.fit_transform(
                    df[col]
                )  # Encode categorical column to numerical
                self.encoders[col] = le  # Store the encoder for later use
        X = df.drop("income", axis=1)  # Features (all columns except 'income')
        y = df["income"]  # Target variable ('income')
        self.X_columns = list(X.columns)  # Store the feature column names
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )  # Split data into training and testing sets
        self.model.fit(X_train, y_train)  # Train the Random Forest model
        y_pred = self.model.predict(X_test)  # Make predictions on the test set
        self.accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        print(f"Model accuracy: {self.accuracy}")  # Print the accuracy
    def predict(self, input_data):
        """Makes a prediction on new input data."""
        df = input_data.copy()  # Create a copy of the input data
        if self.X_columns is None:
            raise RuntimeError(
                "The model has not been trained yet. Please call 'fit' first."
            )
        for col, encoder in self.encoders.items():
            if col in df.columns and col != "income":
                # Use a mapping to handle potentially unseen labels during prediction
                class_mapping = {
                    cls: i for i, cls in enumerate(encoder.classes_)
                }  # Create a mapping from class name to encoded integer
                df[col] = (
                    df[col].map(class_mapping).fillna(-1).astype(int)
                )  # Apply mapping, fill unknown with -1
        # Ensure columns are in the same order as during training
        df = df[self.X_columns]  # Reorder columns to match training data
        prediction_encoded = self.model.predict(
            df
        )  # Get numerical prediction from the model
        prediction_proba = self.model.predict_proba(df)  # Get prediction probabilities
        # Decode the prediction back to the original label (e.g., '>50K')
        prediction_decoded = self.encoders["income"].inverse_transform(
            prediction_encoded
        )  # Convert numerical prediction back to original label
        return (
            prediction_decoded,
            prediction_proba,
        )  # Return decoded prediction and probabilities
