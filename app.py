# import resource

from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


def get_text(url):
    import urlToText
    return urlToText.get_text_data(url)


@app.route('/predict', methods=['POST'])
def predict():
    import tagger
    url = request.form['search']
    text = get_text(url)
    prediction = tagger.getKeyTerms(text)
    return render_template('index.html', prediction_text='Found the following tags {}'.format(prediction))
    # return render_template('index.html', prediction_text='Fetching tags please wait..')


if __name__ == "__main__":
    # resource.setrlimit(resource.RLIMIT_AS, (1000, 1000))
    app.run(host='0.0.0.0', debug=True)
