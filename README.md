# Cardiac_Risk_Predictor

Swagger : https://cardiac-risk-predictor-api.onrender.com/swagger/

{
  "swagger": "2.0",
  "info": {
    "title": "Cardiac Risk Prediction API",
    "version": "1.0",
    "description": "API for making heart disease predictions"
  },
  "basePath": "/",
  "schemes": [ "http", "https" ],
  "consumes": [ "application/json" ],
  "produces": [ "application/json" ],
  "paths": {
    "/predict": {
      "post": {
        "summary": "API to handle the prediction process",
        "parameters": [
          {
            "name": "body",
            "in": "body",
            "required": true,
            "schema": {
              "type": "array",
              "items": {
                "type": "object"
              }
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful response",
            "schema": {
              "type": "object",
              "properties": {
                "prediction_result": {
                  "type": "string",
                  "description": "The prediction result"
                }
              }
            }
          },
          "400": {
            "description": "Bad request",
            "schema": {
              "type": "object",
              "properties": {
                "error_log_prediction": {
                  "type": "string",
                  "description": "Error log for prediction"
                }
              }
            }
          }
        }
      }
    }
  }
}
