from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk.data

def tokenizer(text):
  #make text lowercase?
  #text = text.lower()

  #return pretrained data to determine
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

  # split sentences, then split sentences into tokens
  tokens = [word for sent in sent_detector.tokenize(text) for word in word_tokenize(sent)]

  #remove stopwords and tokens with only one character from tokens
  tokens = [token for token in tokens if ( (len(token) > 1) and (token.lower() not in stopwords.words('english')) )]

  #stem tokens
  stemmer = EnglishStemmer()
  tokens = [stemmer.stem(t) for t in tokens]

  print tokens
  return tokens

if __name__ == '__main__':
  nltk.data.path.append('/home/jocelyn/usb/nltk_data')


  # EXAMPLES OF TOKENIZER

  # simple example
  text = "She looked at her father's arm-chair. Then she sat down."
  print "text: " + text
  print
  tokenizer(text)

  print "----------------------------------"
  print

  # shows how period is left in names but taken out of end of sentences
  text = "Punkt knows that the periods in Mr. Smith and Johann S. Bach do not mark sentence boundaries.  And sometimes sentences can start with non-capitalized words.  i is a good variable name."
  print "text: " + text
  print
  tokenizer(text)

  print "----------------------------------"
  print

  # tokenization of period

  # tokenization of apostrophe

  # tokenization of dates
  text = "Today date is 3/15/2015 and tomorrows date is March 16th 2016. The day after will be 3-17-2016."
  print "text: " + text
  print
  tokenizer(text)

  # tokenization of dash

  # tokenization of comma














