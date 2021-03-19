from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/CNN.html')
def red():
    return render_template('CNN.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
