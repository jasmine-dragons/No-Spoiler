from flask import Flask 
from utils import spoiler_value
app = Flask(__name__)

db = g._database = PyMongo(current_app).db

# will have a json obj as param
@app.route('/create')
def create_post(data):
    '''
    create_post(data)
    data has username, title, description, list of comments
    '''
    val = spoiler_value(data.description)
    if val >= 0.3:
        val['spoiler'] = True
    post_doc = {'username' : data.username, 'description' : data.description, 'comments' : data.comments}
    return db.comments.insert_one(post_doc)
    # return "post created"

# will have a json obj as param
@app.route('/comment')
def comment_post(data):
    val = spoiler_value(data.description)
    if val >= 0.3:
        val['spoiler'] = True
    comment_doc = {'username' : data.username, 'comment' : data.comment, 'spoiler' : val['spoiler']}
    return db.comments.insert_one(comment_doc)
    # return "comment created"

@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"