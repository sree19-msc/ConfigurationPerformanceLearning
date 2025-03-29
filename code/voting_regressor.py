import pandas as pd
import os
import numpy as np
import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7

    results = []

    for current_system in systems:
        datasets_location = f'../datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}')
            
            start_time = time.time()

            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=repeat)
                test_data = data.drop(train_data.index)

                training_X, testing_X = train_data.iloc[:, :-1], test_data.iloc[:, :-1]
                training_Y, testing_Y = train_data.iloc[:, -1], test_data.iloc[:, -1]

                lin_model = LinearRegression()
                xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

                voting_model = VotingRegressor([
                    ('Linear Regression', lin_model),
                    ('XGBoost', xgb_model),
                    ('Random Forest', rf_model)
                ])

                voting_model.fit(training_X, training_Y)
                voting_pred = voting_model.predict(testing_X)

                metrics['MAPE'].append(mean_absolute_percentage_error(testing_Y, voting_pred))
                metrics['MAE'].append(mean_absolute_error(testing_Y, voting_pred))
                metrics['RMSE'].append(np.sqrt(mean_squared_error(testing_Y, voting_pred)))

            end_time = time.time()
            elapsed = end_time - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            avg_mape = np.mean(metrics['MAPE'])
            avg_mae = np.mean(metrics['MAE'])
            avg_rmse = np.mean(metrics['RMSE'])

            print("Voting Regressor Performance:")
            print(f"  - Average MAPE: {avg_mape:.4f}")
            print(f"  - Average MAE:  {avg_mae:.4f}")
            print(f"  - Average RMSE: {avg_rmse:.4f}")
            print(f"  - Time taken:   {minutes} min {seconds} sec")
            print("-" * 60)

            results.append({
                'System': current_system,
                'Dataset': csv_file,
                'MAPE': round(avg_mape, 4),
                'MAE': round(avg_mae, 4),
                'RMSE': round(avg_rmse, 4)
            })

    # Save CSV results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'voting_regressor.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\nAll results have been saved to {output_path}")

if __name__ == "__main__":
    main()

