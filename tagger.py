from spacy import load
from pytextrank import TextRank
from flask import jsonify

# load spacy nlp model for web data - medium size
nlp = load("en_core_web_md", disable=['ner'])

# add textrank to the model pipeline
nlp.add_pipe(TextRank().PipelineComponent, name="textrank", last=True)

def getKeyTerms(text):
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

    # store keyword, it's rank and the associated vector here
    keyWordToScore = {}

    doc = nlp(text)
    phrases = [(' '.join([word for word in str(term.chunks[0]).split() if word not in cachedStopWords]), term.rank) for
               term in doc._.phrases]

    for p in phrases:
        keyWordToScore[str(p[0])] = (str(p[1]), [str(x) for x in nlp(p[0].lower()).vector])

    response = {}
    response["dict"] = keyWordToScore
    return jsonify(response)