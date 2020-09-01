from flask import Flask, request, jsonify, render_template
import tagger

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

def fetchingTags():
    return render_template('index.html', prediction_text='Fetching tags please wait..')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['search']
    fetchingTags()
    prediction = tagger.getTags(url)
    return render_template('index.html', prediction_text='Found the following tags {}'.format(prediction))
    # return render_template('index.html', prediction_text='Fetching tags please wait..')



if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
