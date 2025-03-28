import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def feature_engineering_pipeline(df, target_column='time'):
    """
    A robust feature engineering function that:
      - Checks the target column exists,
      - Dynamically identifies numeric and categorical columns,
      - Imputes missing values,
      - Scales numeric features using StandardScaler,
      - One-hot encodes categorical features (if any),
      - Returns a processed DataFrame with the same 'target_column' appended at the end.
    """
    if target_column == 'time':
        if 'time' not in df.columns and 'throughput' in df.columns:
            df = df.rename(columns={'throughput': 'time'})

    # 2. Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify numeric and categorical features
    numeric_cols = X.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    # If no numeric or categorical columns, raise an error
    if not numeric_cols and not categorical_cols:
        raise ValueError("No valid numeric or categorical columns in this dataset.")

    # Define numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler())
    ])

    # Define categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])

    # Build dynamic column transformer
    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Fit & transform
    X_processed = preprocessor.fit_transform(X)

    # Build final column list
    final_columns = []
    if numeric_cols:
        final_columns.extend(numeric_cols)
    if categorical_cols:
        cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
        final_columns.extend(cat_features.tolist())

    # If we ended up with zero columns, raise an error
    if len(final_columns) == 0:
        raise ValueError("After transformation, zero columns remain.")

    # Construct processed DataFrame
    df_processed = pd.DataFrame(X_processed, columns=final_columns)
    df_processed[target_column] = y.reset_index(drop=True)

    return df_processed
