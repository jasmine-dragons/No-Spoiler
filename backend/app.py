from flask import Flask, request
from flask_cors import CORS
from utils import spoiler_value
from flask import current_app, g
from werkzeug.local import LocalProxy
# from flask_pymongo import PyMongo
from pymongo import MongoClient
from bson.json_util import dumps
app = Flask(__name__)

CORS(app)

# db = g._database = PyMongo(current_app).db

client = MongoClient('mongodb+srv://user:christopher@cluster0.4oqji.mongodb.net/Cluster0?retryWrites=true&w=majority')

db = client.user_inputs

@app.route('/posts', methods=['GET'])
def posts():
    """
    Returns list of all posts in the database.
    """
    # print(db.posts.find({}))
    return dumps(list(db.posts.find({})))

# will have a json obj as param
@app.route('/post', methods=['POST'])
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

@app.route('/post/<uuid: post>/comment', methods=['POST'])
def create_comment(post, data):
    curr_post = db.posts.find(post)

    val = spoiler_value(data.get("content"))
    spoil = True if val >= 0.3 else False

    comment_doc = {
        'username': data.get("username"),
        'content': data.get("content"),
        'spoiler': spoil
    }

    curr_post.comments.appent(comment_doc)

    return db.posts.update(curr_post)


# # will have a json obj as param
# @app.route('/comment', methods=['POST'])
# def comment_post():
#     data = request.get_json()
#     val = spoiler_value(data.get("content"))
#     spoil = False
#     if val >= 0.3:
#         spoil = True
#     comment_doc = {'username' : data.get("username"), 'content' : data.get("content"), 'spoiler' : spoil}
#     result = db.comments.insert_one(comment_doc)
#     return str(result.inserted_id)
#     # return "comment created"

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