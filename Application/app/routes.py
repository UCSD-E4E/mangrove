from app import app
from flask import render_template

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',title='Home')

@app.route('/upload')
def upload():
    return render_template('upload.html',title='Home')

@app.route('/classify')
def classify():
    return render_template('classify.html',title='Home')


