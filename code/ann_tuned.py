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

from feature_engineering import feature_engineering_pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def build_ann(input_dim, dropout_rate=0.2, lr=0.001):
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
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1

    dropout_options = [0.1, 0.2]
    learning_rate_options = [0.001, 0.01]

    results = []

    for current_system in systems:
        datasets_location = f'../datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, "
                  f"Training data fraction: {train_frac}, Number of repeats: {num_repeats}")

            start_time = time.time()

            data_path = os.path.join(datasets_location, csv_file)
            data = pd.read_csv(data_path)

            try:
                data = feature_engineering_pipeline(data, target_column='time')
            except Exception as e:
                print(f"Skipping {csv_file} due to error in feature engineering: {e}")
                continue

            # Hyperparameter search (on one split)
            best_score = float('inf')
            best_params = None

            train_data = data.sample(frac=train_frac, random_state=random_seed)
            test_data  = data.drop(train_data.index)

            training_X = train_data.iloc[:, :-1].values
            training_Y = train_data.iloc[:, -1].values
            testing_X  = test_data.iloc[:, :-1].values
            testing_Y  = test_data.iloc[:, -1].values

            for dropout_rate, lr in product(dropout_options, learning_rate_options):
                model = build_ann(training_X.shape[1], dropout_rate, lr)
                model.fit(training_X, training_Y, epochs=10, verbose=0, batch_size=32)
                preds = model.predict(testing_X, verbose=0).flatten()
                mae = mean_absolute_error(testing_Y, preds)

                if mae < best_score:
                    best_score = mae
                    best_params = (dropout_rate, lr)

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1].values
                training_Y = train_data.iloc[:, -1].values
                testing_X = test_data.iloc[:, :-1].values
                testing_Y = test_data.iloc[:, -1].values

                model = build_ann(training_X.shape[1], dropout_rate=best_params[0], lr=best_params[1])
                model.fit(training_X, training_Y, epochs=10, verbose=0, batch_size=32)
                preds = model.predict(testing_X, verbose=0).flatten()

                mape = mean_absolute_percentage_error(testing_Y, preds)
                mae = mean_absolute_error(testing_Y, preds)
                rmse = np.sqrt(mean_squared_error(testing_Y, preds))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            end_time = time.time()
            total_time = end_time - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)

            avg_mape = np.mean(metrics['MAPE'])
            avg_mae = np.mean(metrics['MAE'])
            avg_rmse = np.mean(metrics['RMSE'])

            print("ANN Tuned Performance:")
            print(f"Average MAPE: {avg_mape:.4f}")
            print(f"Average MAE:  {avg_mae:.4f}")
            print(f"Average RMSE: {avg_rmse:.4f}")
            print(f"Time taken:   {minutes} min {seconds} sec")
            print("-" * 60)

            results.append({
                'System': current_system,
                'Dataset': csv_file,
                'MAPE': round(avg_mape, 4),
                'MAE': round(avg_mae, 4),
                'RMSE': round(avg_rmse, 4)
            })

    # Save to output CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'ann_tuned.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\nAll results have been saved to {output_path}")

if __name__ == '__main__':
    main()
