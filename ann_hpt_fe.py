import os
import time
import warnings
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from itertools import product

# Import your pipeline from a separate file
from feature_engineering import feature_engineering_pipeline

# Suppress TF & Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Build ANN model with a fixed hidden_units=128
def build_ann(input_dim, dropout_rate=0.2, lr=0.001):
    """
    Creates a feed-forward ANN where:
      - hidden units = 128
      - dropout_rate and lr are hyperparameters
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1

    # Hyperparameter options (now only dropout & LR)
    dropout_options = [0.1, 0.2]
    learning_rate_options = [0.001, 0.01]

    for current_system in systems:
        datasets_location = f'D:/ISE_lab2/lab2/datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, "
                  f"Training data fraction: {train_frac}, Number of repeats: {num_repeats}")

            start_time = time.time()  # Start timing

            # Load dataset
            data_path = os.path.join(datasets_location, csv_file)
            data = pd.read_csv(data_path)

            # If 'time' is missing but 'throughput' is there, rename
            if 'time' not in data.columns and 'throughput' in data.columns:
                data.rename(columns={'throughput': 'time'}, inplace=True)

            # Apply feature engineering
            try:
                data = feature_engineering_pipeline(data, target_column='time')
            except Exception as e:
                print(f"Skipping {csv_file} due to error in feature engineering: {e}")
                continue

            # 1) Light hyperparameter search on a single repeat to find best (dropout, lr)
            best_score = float('inf')
            best_params = None

            # We'll do just one data split for this search
            train_data = data.sample(frac=train_frac, random_state=random_seed)
            test_data  = data.drop(train_data.index)

            training_X = train_data.iloc[:, :-1].values
            training_Y = train_data.iloc[:, -1].values
            testing_X  = test_data.iloc[:, :-1].values
            testing_Y  = test_data.iloc[:, -1].values

            # Iterate over possible param combos
            for dropout_rate, lr in product(dropout_options, learning_rate_options):
                model = build_ann(training_X.shape[1], dropout_rate, lr)
                model.fit(training_X, training_Y, epochs=10, verbose=0, batch_size=32)

                preds = model.predict(testing_X, verbose=0).flatten()
                mae  = mean_absolute_error(testing_Y, preds)

                if mae < best_score:
                    best_score = mae
                    best_params = (dropout_rate, lr)

            # 2) Evaluate best hyperparams across 33 repeats
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data  = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1].values
                training_Y = train_data.iloc[:, -1].values
                testing_X  = test_data.iloc[:, :-1].values
                testing_Y  = test_data.iloc[:, -1].values

                # Build model with best hyperparams
                model = build_ann(training_X.shape[1], dropout_rate=best_params[0], lr=best_params[1])
                model.fit(training_X, training_Y, epochs=10, verbose=0, batch_size=32)

                preds = model.predict(testing_X, verbose=0).flatten()

                mape = mean_absolute_percentage_error(testing_Y, preds)
                mae  = mean_absolute_error(testing_Y, preds)
                rmse = np.sqrt(mean_squared_error(testing_Y, preds))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            end_time = time.time()
            total_time = end_time - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)

            print(f"Best Hyperparams: dropout={best_params[0]}, lr={best_params[1]}")
            print(f"Average MAPE: {np.mean(metrics['MAPE']):.4f}")
            print(f"Average MAE:  {np.mean(metrics['MAE']):.4f}")
            print(f"Average RMSE: {np.mean(metrics['RMSE']):.4f}")
            print(f"Time taken:   {minutes} min {seconds} sec")
            print("-" * 60)

if __name__ == '__main__':
    main()
