from spacy import load
from pytextrank import TextRank
from sklearn.cluster import Birch
import urllib.request as urllib2
from readability.readability import Document
from bs4 import BeautifulSoup

# load spacy nlp model for web data - medium size
nlp = load("en_core_web_md", parse=False)

# add textrank to the model pipeline
tr = TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

keyWordToScore = {}

cachedStopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                   "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                   'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                   'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                   'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                   'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                   'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                   'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                   'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                   'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                   'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                   'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                   'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
                   'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                   'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                   "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                   'wouldn', "wouldn't"]


def getTextData(url):
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Referer': 'https://cssspritegenerator.com',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}
    req = urllib2.Request(url, headers=hdr)
    page = urllib2.urlopen(req)
    html = page.read()
    readable_article = Document(html).summary()
    readable_title = Document(html).title()
    # print(readable_title)
    soup = BeautifulSoup(readable_article, features="lxml")
    return soup.text


def getKeyTerms(text, nlp):
    doc = nlp(text)
    phrases = [(' '.join([word for word in str(term.chunks[0]).split() if word not in cachedStopWords]), term.rank) for
               term in doc._.phrases]
    for p in phrases:
        keyWordToScore[str(p[0])] = p[1]
    return [p[0] for p in phrases]


def getTags(url):
    global nlp
    print(str(url))
    # get text data from the url
    text = getTextData(url)

    # fetch the keyterms
    keyTerms = getKeyTerms(text, nlp)

    word_vectors = list(map(lambda x: nlp(x.lower()).vector, keyTerms))
    NUM_CLUSTERS = 8
    kmeans = Birch(n_clusters=NUM_CLUSTERS)
    kmeans.fit(word_vectors)
    labels = kmeans.labels_
    findLabel = lambda x: max(x, key=(lambda key: keyWordToScore[key]))

    output = {}
    for x, y in zip(labels, keyTerms):
        if x in output:
            output[x].append(y)
        else:
            output[x] = [y]

    return "\n"+"\n".join([findLabel(v) for k, v in output.items()])
