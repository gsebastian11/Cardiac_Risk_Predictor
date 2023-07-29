from flask import redirect, render_template, request, session, url_for,jsonify
from app import app
import traceback
import pandas as pd
import numpy as np

#Routes

def configure_routes(app, model, model_columns):
    @app.route('/predict', methods=[ 'POST'])
    def predict():
        """
        API to handle the prediction process.
        """
        if model:
            try:
                json_request = request.json
                print(json_request)

                # load to a dataframe
                df = pd.DataFrame(json_request)
                print(df.head())

                # convert categorical variables into binary variables.
                df = pd.get_dummies(df)
                print(f'After converting categorical values : {df.head()}')

                # reindex to match the columns of the model
                # any missing columns will be replaced with zero.
                df = df.reindex(columns=model_columns, fill_value=0)
                print(f'After reindexing : {df.head()}')

                # predict
                prediction_result = list(model.predict(df))
                print(prediction_result)
                
                return jsonify({'prediction_result': str(prediction_result)})

            except:
                return jsonify({'error_log_prediction': traceback.format_exc()})
        else:
            return ('Model not defined')