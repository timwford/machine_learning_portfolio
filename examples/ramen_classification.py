import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

from typing import List

import numpy as np


class Point:
    def __init__(self, x, y, label=None):
        self.x = x
        self.y = y
        self.label = label
        self.distance = None

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __lt__(self, other):
        return self.distance < other.distance


class KNN:
    def __init__(self, k: int, data: List[Point], debug=False):
        self.k = k
        self.data = data
        self.debug = debug

    def distance(self, x: Point, y: Point):
        if x is None or y is None:
            return "Unable to do anything"

        if x.x is None or x.y is None or y.x is None or y.y is None:
            return "Unable to classify"

        return np.sqrt(np.square(x.x - y.x) + np.square(x.y - y.y))

    def predict(self, input_value: Point):
        unique_labels = []
        for p in self.data:
            p.distance = self.distance(p, input_value)
            if p.label not in unique_labels:
                unique_labels.append(p.label)

        self.data.sort()
        k_nearest = self.data[:self.k]

        dominant_class = {}
        for label in unique_labels:
            dominant_class[label] = 0

        for p in k_nearest:
            dominant_class[p.label] += 1

        if self.debug:
            print(dominant_class)

        return max(dominant_class, key=dominant_class.get)


style_ratings = {
    'Can': 3.5,
    'Cup': 3.4984999999999995,
    'Tray': 3.545138888888889,
    'Bar': 5.0,
    'Bowl': 3.6706860706860707,
    'Box': 4.291666666666667,
    'Pack': 3.7004581151832467
}

country_ratings = {
    'Bangladesh': 3.7142857142857144,
    'Brazil': 4.35,
    'Cambodia': 4.2,
    'Fiji': 3.875,
    'Hong Kong': 3.8018248175182485,
    'Indonesia': 4.067460317460317,
    'Japan': 3.981605113636364,
    'Malaysia': 4.154193548387097,
    'Mexico': 3.73,
    'Myanmar': 3.9464285714285716,
    'Sarawak': 4.333333333333333,
    'Singapore': 4.126146788990826,
    'South Korea': 3.7905537459283383,
    'Taiwan': 3.665401785714286,
    'United States': 3.75,
    'Australia': 3.1386363636363637,
    'Canada': 2.2439024390243905,
    'China': 3.4218934911242602,
    'Colombia': 3.2916666666666665,
    'Dubai': 3.5833333333333335,
    'Estonia': 3.5,
    'Finland': 3.5833333333333335,
    'Germany': 3.638888888888889,
    'Ghana': 3.5,
    'Holland': 3.5625,
    'Hungary': 3.611111111111111,
    'India': 3.3951612903225805,
    'Nepal': 3.5535714285714284,
    'Netherlands': 2.4833333333333334,
    'Nigeria': 1.5,
    'Pakistan': 3.0,
    'Philippines': 3.3297872340425534,
    'Poland': 3.625,
    'Sweden': 3.25,
    'Thailand': 3.3848167539267022,
    'UK': 2.9971014492753625,
    'USA': 3.457043343653251,
    'Vietnam': 3.187962962962963
}

average = 3.6546759798214974


def generate_ramen_point(country: str, style: str):
    try:
        style_value = style_ratings[style]
        country_value = country_ratings[country]
        return Point(style_value, country_value)
    except KeyError as e:
        print(f"No data on input {str(e)}")

def can_classify_ramen(country: str, style: str) -> bool:
    try:
        style_value = style_ratings[style]
        country_value = country_ratings[country]
        return True
    except KeyError as e:
        print(f"No data on input {str(e)}")
        return False


class EatTheRamen:
    def __init__(self):
        df = pd.read_csv("final_ramen.csv")

        style_points = df['Style_value_jitter']
        country_points = df['Country_value_jitter']
        labels = df['label'].astype(str)

        points = []
        for i in range(len(style_points)):
            points.append(Point(style_points[i], country_points[i], labels[i]))

        model = KNN(15, points)

        self.model = model

    test_country = 'USA'
    test_style = 'Pack'


def get_style_docs(value, description: str):
    ratings = dict(sorted(value.items(), key=lambda item: item[1]))
    ratings_keys = [str(x) for x in list(ratings.keys())]
    ratings_values = ["{:.2f}".format(x) for x in list(ratings.values())]

    doc = "\n" + description + " (Key, average review score)\n\n"
    doc += "Above Average:\n"
    for i in range(len(ratings_keys)):
        if float(ratings_values[i]) > average:
            doc += "'" + ratings_keys[i] + "': " + ratings_values[i] + "\n"

    doc += "\nBelow Average:\n"
    for i in range(len(ratings_keys)):
        if float(ratings_values[i]) <= average:
            doc += "'" + ratings_keys[i] + "': " + ratings_values[i] + "\n"

    return doc


# FastAPI App and the Model
app = FastAPI(title="Is the ramen good?",
              description="Provide how your ramen will be served/how it was packaged, "
                          "selected from the following list ():\n\n" +
                          get_style_docs(style_ratings, 'Ratings for Ramen packaging/serving style') +
                          "\n\nand what country your ramen was made in from these options:\n\n" +
                          get_style_docs(country_ratings, 'Ratings for Ramen made in specific countries'))

eatTheRamenator = EatTheRamen()


# Pydantic Models
class EatTheRamenSchema(BaseModel):
    eat_the_ramen: str


class RamenInput(BaseModel):
    country: str
    style: str


@app.get("/")
async def root():
    return {
        "Localhost": 'http://127.0.0.1:5000/docs#/default/should_i_eat_the_ramen_ramen_get',
        "Hosted": 'https://homemetricsdev.uc.r.appspot.com/docs#/default/should_i_eat_the_ramen_ramen_post'
    }


@app.post("/ramen",
          response_model=EatTheRamenSchema,
          description="Press 'Try it out' and then enter where your ramen is from "
                      "and how its served/packaged from the list above! Then press"
                      "'Execute' and let the algorithm determine whether the ramen"
                      "is likely to be tasty 🍜"
          )
async def should_i_eat_the_ramen(ramen: RamenInput):
    if can_classify_ramen(ramen.country, ramen.style):
        result = eatTheRamenator.model.predict(generate_ramen_point(ramen.country, ramen.style))
    else:
        result = generate_ramen_point(ramen.country, ramen.style)

    return EatTheRamenSchema(
        eat_the_ramen=result
    )


if __name__ == "__main__":
    uvicorn.run("ramen_classification:app", host="127.0.0.1", port=5000, log_level="info")
