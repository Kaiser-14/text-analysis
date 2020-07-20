import unidecode as unidecode
import matplotlib.pyplot as plt
import re

from num2words import num2words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import string

hamlet = open('Hamlet.txt', encoding="utf8").read().split('\n')
# When printing, we check that there are many things to do with it, for example, remove the first 8 characters for each
# line
# print(hamlet)

# Necessary to take only the text itself, not the description, so I consider only from line 333 to 7059
hamlet = hamlet[332:7059]
# print(hamlet)

# Mapping of names
characters_contractions_mapping = {
    "King": "King", "Ham": "Hamlet", "Pol": "Polonius", "Hor": "Horatio", "Laer": "Laertes", "Volt": "Voltimand",
    "Cor": "Cornelius", "Ros": "Rosencrantz", "Guil": "Guildenstern", "Osr": "Osric", "Gent": "Gentleman",
    "Mar": "Marcellus", "Ber": "Bernardo", "Fran": "Francisco", "Rey": "Reynaldo", "Clown": "Clowns",
    "Fort": "Fortinbras", "Capt": "Captain", "Ghost": "Ghost", "Queen": "Queen", "Oph": "Ophelia"}

# In order to preprocess all the data, I try to do it in only one loop
# Create tokens for each sentence
hamlet_processed = list()
porter = PorterStemmer()
lem = WordNetLemmatizer()


contractions_re = re.compile('(%s)' % '|'.join(characters_contractions_mapping.keys()))


def expand_contractions(s, characters_contractions_mapping=characters_contractions_mapping):
    def replace(match):
        return characters_contractions_mapping[match.group(0)]
    return contractions_re.sub(replace, s)


for sentence in hamlet:
    # Remove extra numeration columns in the text
    sentence = sentence[8:]
    # Remove empty lines
    if sentence == "": continue
    # Tokenize Hamlet
    tokens = word_tokenize(sentence)
    # Expand contractions characters
    tokens.append(expand_contractions(str(tokens)))
    # Remove punctuation
    tokens = [re.sub(r'[^a-zA-Z0-9]', r' ', token) for token in tokens]
    # Remove accents
    tokens = [unidecode.unidecode(token) for token in tokens]
    # Remove blanks, convert to lower case and replace numbers by words
    tokens = [num2words(int(token.strip().lower()), lang='en') if token.isdigit()
              else token.strip().lower()
              for token in tokens]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Remove blanks between sentences
    tokens = [token for token in tokens if token != '']
    tokens = [lem.lemmatize(word) for word in tokens]
    # tokens = porter.stem(tokens)

    hamlet_processed.append(tokens)

print(hamlet_processed)
# print("Length of the file processed",len(hamlet_processed))

# Term Frequency distribution to show the most used words
flat_list = [item for sublist in hamlet_processed for item in sublist]
# print(flat_list)
fdist = FreqDist(flat_list)
fdist.plot(30, cumulative=False)
plt.show()

# Bag of words
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1, 1), analyzer="word")
train_bow = bow.fit_transform(flat_list)
# print(train_bow)

# Discover the main character, counting the names of each main character
# First, insert the main characters
main_char = list(characters_contractions_mapping.keys())
#main_char = {'Hamlet', 'Claudius', 'Gertrude', 'Ghost', 'Polonius', 'Laertes', 'Ophelia', 'Horatio',
#             'Rosencrantz', 'Guildenstern', 'Fortinbras', 'Voltimand', 'Cornelius', 'Osric', 'Marcellus'}

flat_list1 = [item for sublist in hamlet for item in sublist]
hamlet_text = ''.join(map(str, flat_list1))
#print(flat_list1)
#print(hamlet_text)

char_count = dict((x, 0) for x in main_char)
for w in re.findall(r"\w+", hamlet_text):
    if w in char_count:
        char_count[w] += 1
# print(wordcount)
plt.bar(range(len(char_count)), list(char_count.values()), align='center')
plt.xticks(range(len(char_count)), list(char_count.keys()), rotation=90)
plt.ylabel('counts')
plt.show()

# Word2Vec
word2vec = Word2Vec(hamlet_processed, min_count=2)
vocabulary = word2vec.wv.vocab
# print(vocabulary)
# Similar words
word = "king"
sim_words = word2vec.wv.most_similar(word)
print("Most related words to: ", word)
for word in sim_words:
  print(word[0], word[1])
