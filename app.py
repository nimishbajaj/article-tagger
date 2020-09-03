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
    keyTerms = tagger.getKeyTerms(text)

    import requests

    url = "https://xacyumx10a.execute-api.us-east-2.amazonaws.com/dev/cluster"

    payload = keyTerms
    headers = {
        'Content-Type': "application/json",
        'User-Agent': "PostmanRuntime/7.19.0",
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Postman-Token': "c95005fc-507b-4e95-8586-32241a13b6b2,79b55e9d-7b64-4809-91ef-24cd673f12ae",
        'Host': "xacyumx10a.execute-api.us-east-2.amazonaws.com",
        'Accept-Encoding': "gzip, deflate",
        'Content-Length': "1413790",
        'cache-control': "no-cache"
    }

    response = requests.request("POST", url, data=payload, headers=headers)

    print(response.text)

    # print(response.text)
    return render_template('index.html', prediction_text='Found the following tags {}'.format(response.text))
    # return render_template('index.html', prediction_text='Fetching tags please wait..')


if __name__ == "__main__":
    # resource.setrlimit(resource.RLIMIT_AS, (1000, 1000))
    app.run(host='0.0.0.0', debug=True)
