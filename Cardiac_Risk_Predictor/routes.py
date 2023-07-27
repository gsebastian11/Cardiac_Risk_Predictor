from flask import redirect, render_template, request, session, url_for
from app import app

#Routes
@app.route('/')
def home():
    """
    Renders the home page.
    """

@app.route('/predict', methods=[ 'POST'])
def predict():
    """
    Handles the prediction process.
    """