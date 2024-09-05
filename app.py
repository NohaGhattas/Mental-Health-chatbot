import random
from flask import Flask, request, jsonify, make_response, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api, Resource, reqparse
from werkzeug.exceptions import BadRequest
import os
import datetime
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import pickle

app = Flask(__name__)
api = Api(app)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  # SQLite database
db = SQLAlchemy(app)

class PostModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), unique=True, nullable=False)
    text = db.Column(db.Text(), nullable=True)
    images = db.Column(db.JSON(), nullable=True)
    video_link = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime(), nullable=False)
    updated_at = db.Column(db.DateTime(), nullable=True)

    def __repr__(self):
        return f"Post(title={self.title})"

    def as_dict(self):
        return {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }

if os.path.exists('instance/database.db'):
    print('Database already exists')
else:
    with app.app_context():
        db.create_all()
    print('Database created successfully')

class Posts(Resource):
    def post(self):
        content_type_version = request.headers.get('Content-Type')

        posts_post_args = reqparse.RequestParser()
        posts_post_args.add_argument("title", type=str, required=True, help="Post title is required")

        if content_type_version == 'application/vnd.blog.com.v1+json':
            posts_post_args.add_argument("text", type=str, required=False)
        elif content_type_version == 'application/vnd.blog.com.v2+json':
            posts_post_args.add_argument("text", type=str, required=True)
        else:
            return make_response(jsonify({'error': 'Unsupported version'}), 400)

        posts_post_args.add_argument("images", type=str, required=False, action="append")
        posts_post_args.add_argument("video_link", type=str, required=False)

        try:
            args = posts_post_args.parse_args()
            post = PostModel(
                title=args["title"],
                text=args.get("text"),
                images=args.get("images"),
                video_link=args.get("video_link"),
                created_at=datetime.datetime.now(),
            )
            db.session.add(post)
            db.session.commit()

            accept_version = request.headers.get('Accept')

            if accept_version == 'application/vnd.blog.com.v1+json':
                return make_response(jsonify(post.as_dict()), 201)
            elif accept_version == 'application/vnd.blog.com.v2+json':
                return make_response(jsonify({"post": post.as_dict()}), 201)
            else:
                return make_response(jsonify({'error': 'Unsupported version'}), 400)
        except Exception as e:
            return make_response(jsonify({"error": f"An error occurred: {str(e)}"}), 500)

    def get(self):
        posts = PostModel.query.all()
        posts_as_json = [post.as_dict() for post in posts]
        return make_response(jsonify(posts_as_json), 200)

class Post(Resource):
    def get(self, post_id):
        post = PostModel.query.filter_by(id=post_id).first()
        if not post:
            return make_response(jsonify({"error": f"Post with ID {post_id} not found"}), 404)
        return make_response(jsonify(post.as_dict()), 200)

    def put(self, post_id):
        v = request.args.get('v')
        api_version = v if v in ('1', '2') else '1'

        post_put_args = reqparse.RequestParser()
        post_put_args.add_argument("title", type=str, required=True, help="Post title is required")

        if api_version == '1':
            post_put_args.add_argument("text", type=str, required=False)
        elif api_version == '2':
            post_put_args.add_argument("text", type=str, required=True)

        post_put_args.add_argument("images", type=str, required=False, action="append")
        post_put_args.add_argument("video_link", type=str, required=False)

        try:
            args = post_put_args.parse_args()
            post = PostModel.query.filter_by(id=post_id).first()
            if not post:
                return make_response(jsonify({"error": f"Post with ID {post_id} not found"}), 404)
            else:
                for arg in args:
                    if arg in post.__table__.columns:
                        setattr(post, arg, args[arg])
                post.updated_at = datetime.datetime.now()
                db.session.add(post)
                db.session.commit()

                if api_version == '1':
                    return make_response(jsonify(post.as_dict()), 200)
                elif api_version == '2':
                    return make_response(jsonify({"post": post.as_dict()}), 200)
        except Exception as e:
            return make_response(jsonify({"error": f"An error occurred: {str(e)}"}), 500)

    def delete(self, post_id):
        if not PostModel.query.filter_by(id=post_id).first():
            return make_response(jsonify({"error": f"Post with ID {post_id} not found"}), 404)

        PostModel.query.filter_by(id=post_id).delete()
        db.session.commit()
        return make_response(jsonify({"message": f"Post with ID {post_id} deleted successfully"}), 200)

api.add_resource(Posts, "/api/posts")
api.add_resource(Post, "/api/posts/<int:post_id>")

# Chatbot setup
lemmatizer = WordNetLemmatizer()

try:
    with open('D:\Chatbot\chatbot\Include\intents.json') as f:
        intents = json.load(f)
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model("chatbot_model.h5")
except Exception as e:
    print(f"Error loading resources: {e}")
    exit(1)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            index = words.index(w)
            bag[index] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    responses = [i['responses'] for i in intents_json['intents'] if i['tag'] == tag]
    if responses:
        return random.choice(responses[0])
    return "Sorry, I didn't find a response."

@app.route("/chat", methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        if not user_message:
            return make_response(jsonify({'error': 'No message provided'}), 400)
        ints = predict_class(user_message)
        response = get_response(ints, intents)
        return make_response(jsonify({'response': response}))
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
