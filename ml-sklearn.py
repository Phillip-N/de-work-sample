import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import logging

try:
    # Set up logging
    logging.basicConfig(filename='training-sklearn.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the dataset
    data = pd.read_parquet('solution-2.parquet')
    data['Date'] = pd.to_datetime(data['Date'])

    # Preprocess the data
    data.set_index('Date', inplace=True)
    data.dropna(inplace=True)

    # Select features and target
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    X = data[features]
    y = data[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6, n_jobs=-1, verbose=10)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Log training metrics
    logging.info(f'Mean Absolute Error: {mae}')
    logging.info(f'Mean Squared Error: {mse}')

    # Save the trained model
    pickle.dump(model, open('ml_model.pkl', 'wb'))

    logging.info('Training completed.')

except Exception as e:
    logging.error(f'An error occurred: {str(e)}')