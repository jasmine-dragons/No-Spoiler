from flask import Flask
import spoiler_value from 'utils.py'

app = Flask(__name__)

# will have a json obj as param
@app.route('/create')
def create_post(data):
    val = spoiler_value(data.description)
    if val >= 0.3:
        val['spoiler'] = True
    return "post created"

# will have a json obj as param
@app.route('/comment')
def comment_post(data):
    val = spoiler_value(data.description)
    if val >= 0.3:
        val['spoiler'] = True
    return "comment created"

@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"