import sys
from catboost import CatBoostClassifier
import pandas as pd
from flask import Flask, request, flash, redirect, url_for

app = Flask(__name__)


@app.route('/')
def hello_world():
    print('This is standard output', file=sys.stdout)
    return 'Hello World!'


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        from_file = CatBoostClassifier()
        model = from_file.load_model("model")
        data = pd.read_csv(uploaded_file, header=None)
        data = pd.DataFrame(data).to_numpy()

        res = ''
        print(len(data[0]), len(data), file=sys.stdout)
        if len(data) == 1 and len(data[0]) == 188:
            res = model.predict(data)
        else:
            return 'incorrect data', 400
    return ' '.join(map(str, res[0]))


if __name__ == '__main__':
    app.run()
