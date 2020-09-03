import urllib.request as urllib2
from readability.readability import Document
from bs4 import BeautifulSoup


def get_text_data(url):
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
