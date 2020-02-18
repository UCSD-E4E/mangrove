import os
import sys
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
#from flask_wtf import FlaskForm
from wtforms import Form, StringField, PasswordField, BooleanField, SubmitField,validators
from wtforms.validators import DataRequired


app = Flask(__name__)
app.config['SECRET_KEY'] = "it is a secret"
training_file = ""

class LoginForm(Form):
    file_path = StringField('File Path', [validators.DataRequired()])
    submit = SubmitField('Upload')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',title='Home')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = LoginForm()
    print("Form is created", file=sys.stderr)
    if request.method == 'POST' and form.validate():
        print("In valide on submit", file=sys.stderr)
        training_file = form.file_path.data
        flash('File path is {}'.format(training_file))
        return redirect('/classify.html')
    return render_template('upload.html', title='Sign In', form=form)

@app.route('/classify')
def classify():
	return render_template('classify.html')



if __name__ == '__main__':
    app.run(debug=True)


