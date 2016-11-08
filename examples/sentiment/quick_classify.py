import deep_learning_keras
import spacy
import pathlib

MODEL_DIR=pathlib.Path('sentiment_model')
MAX_LENGTH=100

def create_pipeline(nlp, analyser):
    '''
    This could be a lambda, but named functions are easier to read in Python.
    '''
    return [nlp.tagger, nlp.parser, analyser]

nlp = spacy.load('en')
print('Loaded Spacy')

analyser = deep_learning_keras.SentimentAnalyser.load(MODEL_DIR, nlp, max_length=MAX_LENGTH)

nlp.pipeline = create_pipeline(nlp, analyser)

# ---------------------------------------------------------------------
# At this point in the code, you have an nlp instance that will assign
# an attribute 'sentiment' to each doc, using the indicated keras model
# This should be used as self.nlp in a pipelet to use sentiment analysis
# self.nlp = nlp

def quick_evaluate(texts, nlp, max_length=100):

    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):

        print('Got Sentiment {} for document {}'.format(doc.sentiment, doc.text))

print('Running Quick Eval 1')
quick_evaluate([u'This is something really bad'], nlp)
print('Running Quick Eval 2')
quick_evaluate([u'This is much better than I expected it to be'], nlp)
print('Running Quick Eval 3')
quick_evaluate([u'Whoa this could be a game changer'], nlp)
print('Running Quick Eval 4')
quick_evaluate([u'Damn we are really on to something here'], nlp)
