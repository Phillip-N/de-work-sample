from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from ml_predictions import sklearn_predict

app = Flask(__name__)
api = Api(app)

class PredictStockVolume(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('vol_moving_avg', type = float, required=True, location='args')
        self.reqparse.add_argument('adj_close_rolling_med', type = float, required=True, location='args')
        super(PredictStockVolume, self).__init__()

    def get(self):
        args = self.reqparse.parse_args()
        if args['vol_moving_avg'] < 0:
            return "vol_moving_avg must be a non-negative number"
        else:
            prediction = sklearn_predict(args['vol_moving_avg'], args['adj_close_rolling_med'])[0]
            return prediction

api.add_resource(PredictStockVolume, '/predict')

if __name__ == '__main__':
    app.run(debug=True)