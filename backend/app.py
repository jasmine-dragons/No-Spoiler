from flask import Flask

app = Flask(__name__)

# will have a json obj as param
@app.route('/create')
def create_post():
    return "post created"    

# will have a json obj as param
@app.route('/comment')
def comment_post():
    return "comment created"





@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"