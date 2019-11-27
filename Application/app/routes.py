import flask
from app import app
from flask import render_template
from app.filepath import FilePathForm

#import ../CNN Development/autoclass


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',title='Home')

@app.route('/upload')
def upload():
    file_ = FilePathForm()
    return render_template('upload.html',title='Home',file = file_)
    #return render_template('upload.html',title='Home')

@app.route('/classify')
def classify():
    return render_template('classify.html',title='Home')


