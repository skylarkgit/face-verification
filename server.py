from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import werkzeug
import face
import pprint
from matplotlib import pyplot
import facenet.src.facenet as facenet
from keras.models import load_model
import os
import tensorflow as tf
from flask_cors import CORS, cross_origin

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = load_model('facenet_keras.h5')
model._make_predict_function()
graph = tf.get_default_graph()

class Face(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('source', type=werkzeug.datastructures.FileStorage, location='files')
        parse.add_argument('target', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        sourceImage = face.extract_face(args['source'])
        targetImage = face.extract_face(args['target'])
        distance = None
        with graph.as_default():
            sourceEmbedding = face.get_embedding(model, sourceImage)
            targetEmbedding = face.get_embedding(model, targetImage)
            distance = facenet.distance(sourceEmbedding, targetEmbedding)
        return {'distance': distance.tolist()}

api.add_resource(Face, '/verify')

app.run(port='5002')