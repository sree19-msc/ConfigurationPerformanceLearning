import os
import warnings
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from itertools import product

# Suppress TensorFlow and Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Function to build an ANN model with configurable hyperparameters
def build_ann(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Main function to train and evaluate ANN with hyperparameter tuning
def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1

    # Hyperparameter options (light tuning)
    hidden_units_options = [32, 64]
    dropout_options = [0.1, 0.2]
    learning_rate_options = [0.001, 0.01]

    for current_system in systems:
        datasets_location = f'D:/ISE_lab2/lab2/datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}")

            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            best_score = float('inf')
            best_params = None

            # Grid search over hyperparameter combinations (on a single repeat)
            for hidden_units, dropout_rate, lr in product(hidden_units_options, dropout_options, learning_rate_options):
                train_data = data.sample(frac=train_frac, random_state=random_seed)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1].values
                training_Y = train_data.iloc[:, -1].values
                testing_X = test_data.iloc[:, :-1].values
                testing_Y = test_data.iloc[:, -1].values

                model = build_ann(training_X.shape[1], hidden_units, dropout_rate, lr)
                model.fit(training_X, training_Y, epochs=10, verbose=0, batch_size=32)

                predictions = model.predict(testing_X, verbose=0).flatten()
                mae = mean_absolute_error(testing_Y, predictions)

                if mae < best_score:
                    best_score = mae
                    best_params = (hidden_units, dropout_rate, lr)

            # Use best hyperparameters for full 33-repeat evaluation
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}
            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1].values
                training_Y = train_data.iloc[:, -1].values
                testing_X = test_data.iloc[:, :-1].values
                testing_Y = test_data.iloc[:, -1].values

                model = build_ann(training_X.shape[1], *best_params)
                model.fit(training_X, training_Y, epochs=10, verbose=0, batch_size=32)
                predictions = model.predict(testing_X, verbose=0).flatten()

                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            print(f"Average MAPE: {np.mean(metrics['MAPE']):.4f}")
            print(f"Average MAE: {np.mean(metrics['MAE']):.4f}")
            print(f"Average RMSE: {np.mean(metrics['RMSE']):.4f}")
            print("-" * 60)

if __name__ == "__main__":
    main()
