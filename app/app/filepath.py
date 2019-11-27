from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class FilePathForm(FlaskForm):
    file_path = StringField('Paste File Path', validators=[DataRequired()])
    confirm = SubmitField('Confirm')