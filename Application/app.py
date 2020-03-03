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
    file_path = StringField('File Path',validators=[DataRequired()])
    submit = SubmitField('Upload')

def file_exists():
    path = request.form.get('file_path')
    if os.path.exists(path):
        print("File exists!",file=sys.stderr)
        return True
    print("File not exists!",file=sys.stderr)
    return False
    

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',title='Home')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    '''
    form = LoginForm()
    #print("Form is created", file=sys.stderr)
    form.validate()
    print("Form's errors is {}".format(form.errors), file=sys.stderr)
    print("Form's file_path is {}".format(form.file_path.data),file=sys.stderr)

    if request.method == 'POST' and form.validate():
        print("In validate on submit", file=sys.stderr)
        
        training_file = form.file_path.data
        print('File path is {}'.format(training_file),file=sys.stderr)

        return render_template('classify.html')

    return render_template('upload.html', title='Sign In', form=form)
    '''
    
    if request.method == 'POST' and file_exists():  #this block is only entered when the form is submitted
        training_file = request.form.get('file_path')
        print(training_file,file=sys.stderr)
        return render_template('classify.html',file_path = training_file)

    return render_template('upload.html', title='Sign In')

@app.route('/classify')
def classify():
	return render_template('classify.html')



if __name__ == '__main__':
    app.run(debug=True)


