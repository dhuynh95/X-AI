import pandas as pd
import sklearn.linear_model
import numpy as np
from flask_restplus import Resource, Api, reqparse
from flask import Flask

# First we load the data and train our model
data = pd.read_csv("./data/ratWeight.csv")
rat = data.loc[data.id == "B38602"]

x = data.week.values.reshape(-1,1)
y = data.weight.values.reshape(-1,1)

# We fit the model
model = sklearn.linear_model.LinearRegression()
model.fit(x,y)

# We create a parser to get the week argument
parser = reqparse.RequestParser()
parser.add_argument("week", type = int, location = "args", help = "Age of the rat in weeks")

# We create a Flask Resource which serves as an endpoint for the requests
class RatWeight(Resource):
    def get(self):
        # We first get the args
        args = parser.parse_args()

        # We get the week and reshape it to feed it to our model
        week = args["week"]
        week = np.array(week).reshape(-1,1)

        # We get our model prediction
        weight = model.predict(week)

        # We return it
        output = f"Your rat at age {week[0][0]} week(s) should weigh {weight[0][0]} grams"

        return output

# We create the Flask app and API
app = Flask(__name__)
api = Api(app)

# We bind the endpoint
api.add_resource(RatWeight, "/rat_weight")

# We run the app
if __name__ == "__main__":
    app.run()