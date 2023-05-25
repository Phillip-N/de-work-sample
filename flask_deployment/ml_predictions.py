#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle

# Prediction implementation with sklearn
def sklearn_predict(vma, acrm):
    filename = 'ml_model.pkl'

    loaded_model = pickle.load(open(filename, 'rb'))

    # Create a DataFrame with the inputs for prediction
    input_data = pd.DataFrame({'vol_moving_avg': [vma], 'adj_close_rolling_med': [acrm]})

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data)

    return prediction