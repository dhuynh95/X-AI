import pandas as pd
import sklearn.linear_model
import numpy as np
from flask_restplus import Resource, Api, reqparse
from flask import Flask

# First we load the data and train our model
data = pd.read_csv("./data/ratWeight.csv")
rat = data.loc[data.id == "B38602"]

x = rat.week.values.reshape(-1,1)
y = rat.weight.values.reshape(-1,1)

# We fit the model
model = sklearn.linear_model.LinearRegression()
model.fit(x,y)

# We create a parser to get the week argument
parser = reqparse.RequestParser()
parser.add_argument("week", type = int, required = True, help = "Age of the rat in weeks")

# We create 
class RatWeight(Resource):
    def post(self):
        args = parser.parse_args()

        week = args["week"]
        week = np.array(week).reshape(-1,1)

        weight = model.predict(week)

        output = f"Your rat at age {week} should weigh {weight}"

        return output

app = Flask(__name__)
api = Api(app)

api.add_resource(RatWeight, "/rat_weight")

if __name__ == "__main__":
    app.run()