from flask import Flask, request
from flask_cors import CORS
from utils import spoiler_value
from flask import current_app, g
from werkzeug.local import LocalProxy
# from flask_pymongo import PyMongo
from pymongo import MongoClient
app = Flask(__name__)

CORS(app)

# db = g._database = PyMongo(current_app).db

client = MongoClient('mongodb+srv://user:christopher@cluster0.4oqji.mongodb.net/Cluster0?retryWrites=true&w=majority')

db = client.user_inputs

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
    post_doc = {'username' : data.username, 'title' : data.title, 'description' : data.description, 'comments' : data.comments}
    return db.posts.insert_one(post_doc)
    # return "post created"

# will have a json obj as param
@app.route('/comment')
def comment_post(data):
    val = spoiler_value(data.description)
    if val >= 0.3:
        val['spoiler'] = True
    comment_doc = {'username' : data.username, 'content' : data.content, 'spoiler' : val['spoiler']}
    return db.comments.insert_one(comment_doc)
    # return "comment created"

# will have a json obj as param
@app.route('/test')
def test_spoil():
    '''
    test_spoil()
    '''
    text = request.args.get('text')
    val = spoiler_value(text)
    return str(val)
    # return "post created"


@app.route('/getposts')
def getposts():
    """
    Returns list of all posts in the database.
    """
    # print(db.posts.find({}))
    return str(list(db.posts.find({})))

@app.route('/signin')
def signin():
    """
    Returns list of all posts in the database.
    """
    username = request.args.get('username')
    password = request.args.get('password')
    return "success: " + str(username) + " " + str(password)

@app.route('/')
def get_posts():
    return "<p>Hello, World!</p>"