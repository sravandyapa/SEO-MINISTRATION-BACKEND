from flask import Flask
from flask_restful import Resource, Api, reqparse
from modules.model import MLModel
from flask_cors import CORS
import logging

app = Flask(__name__)
api = Api(app)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class ModelLink(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('url', type=str, required=True, help="This is a mandatory field")
        data = parser.parse_args()
        
        result = MLModel.test_link(data['url']);
        
        tags=[]
        
        for tag in result[0]:
            tags.append(tag)
            
        return {'tags':tags},200,\
    { 'Access-Control-Allow-Origin': '*', \
      'Access-Control-Allow-Methods' : 'POST' }

class ModelContent(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str, required=True, help="This is a mandatory field")
        data = parser.parse_args()
        
        result = MLModel.test_para(data['content']);
        
        tags=[]
        
        for tag in result[0]:
            tags.append(tag)
        
        return {'tags':tags},200,\
    { 'Access-Control-Allow-Origin': '*', \
      'Access-Control-Allow-Methods' : 'POST' }

api.add_resource(ModelLink,"/link")
api.add_resource(ModelContent,"/content")

logger=logging.getLogger()

@app.before_first_request
def firstRequest():
    logger.info("Model started training")
    MLModel.train()
    logger.info("Model trained")

app.run(port=12345)
        
