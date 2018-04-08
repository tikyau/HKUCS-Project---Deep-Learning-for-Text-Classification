import sys

from predictor import Predictor
from flask import Flask, render_template, request


app = Flask(__name__)
_predictor = None

def _init_app(vocabulary, labels, model):
    global _predictor
    _predictor = Predictor(vocabulary, labels, model)
    return app

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        inp = request.form['input']
        return render_template(
            'result.html',
            inp=validate(inp),
            res=_predictor.predict(inp)
        )


def validate(inp):
    # TODO
    return inp


if __name__ == '__main__':
    _init_app(sys.argv[1], sys.argv[2], sys.argv[3]).run(host='0.0.0.0')
