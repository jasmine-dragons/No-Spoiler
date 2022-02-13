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

def spoiler_val():
    result = request.form.to_dict(flat=True)
    sentence = result.get("sentence")
    spoil_val = utils.spoiler_value(sentence)
    result["spoil_val"] = spoil_val
    return render_template("index.html", result=result)




@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"