# encoding=utf8
from nltk.corpus import machado, mac_morpho, stopwords
from nltk.stem import RSLPStemmer
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
import numpy
import re
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB


def normalize_text(text):
    '''
    Um exemplo de formas diferentes para normalizar um texto
    usando ferramentas do NLTK

    param: text Uma string com texto que será processado
    '''
    text = text.decode('utf8')
    stemmer = RSLPStemmer()  # Carregando um radicalizador para o PT-BR
    print(text)
    for sent in sent_tokenize(text):
        # Testando formas de tokenização
        tokens = wordpunct_tokenize(sent)
        print(sent)
        print('   wordpunct: \t%s' % ' '.join(tokens))

        tokens = word_tokenize(sent)
        print('        word: \t%s' % ' '.join(tokens))

        # Removendo stopwords
        tokens = remove_stopwords(tokens)
        print('  -stopwords: \t%s' % ' '.join(tokens))

        # Radicalizando as palavras restantes
        tokens = [stemmer.stem(t) for t in tokens]
        print('radicalizado: \t%s' % ' '.join(tokens))

        print('')


def remove_stopwords(tokens):
    '''
    Um método para remoção de stopwords

    param: tokens Uma lista com as palavras de um texto/sentença
    '''
    return [
        t for t in tokens
        if t not in stopwords.words('portuguese')]


def train_tagger():
    '''
    Um exemplo de treinamento de um etiquetador sintático usando
    um modelo de tri-gramas baseado em probabilidades.

    Um etiquetador sintático identifica quais a classe de uma palavra
    Ex.: Isso é um teste = Isso-PROSUB é-V um-ART teste-N
    Preposição Verbo Artigo Substantivo
    '''

    # Carregando um conjunto de dados em português que possui
    # sentenças manualmente identificadas
    data = [
        [(w, re.split('[|-]', tag)[0]) for w, tag in sent]
        for sent in mac_morpho.tagged_sents()]

    # Classe sintática padrão. N siginifica Nome/substantivo
    tagger0 = DefaultTagger('N')
    print('train unigram')
    tagger1 = UnigramTagger(data, backoff=tagger0)
    print('training bigram')
    tagger2 = BigramTagger(data, backoff=tagger1)
    print('training trigram')
    return TrigramTagger(data, backoff=tagger2)


###############################################################################
# Um exemplo de classificação textual
###############################################################################
def read_machado():
    '''
    Esse método carrega um corpus com obras de Machado de Assis
    com as respectivas categorias
    '''
    labels = machado.categories()
    data = list()
    for l in labels:
        # Capturando todos os textos da categoria l
        text_ids = machado.fileids(categories=l)
        data.extend([(machado.raw(fileids=tid), l) for tid in text_ids])
    return data


class TextProcessor(object):
    '''
    Uma classe para processar os textos
    '''

    def __init__(self):
        self.stemer = RSLPStemmer()
        # Podemos definir mais alguns processo aqui, como um tagger, ...

    def process(self, document):
        # lendo todas as palavras do documento
        tokens = [
            re.sub('\d+[,.]?\d+', '<NUM>', t)
            for sent in sent_tokenize(document)
            for t in word_tokenize(sent)]

        return [
            t if t.startswith('<') else self.stemer.stem(t)
            for t in tokens
            if t not in stopwords.words('portuguese')]


def extract_features(documents):
    '''
    Extrai as features dos textos
    '''
    processor = TextProcessor()
    # Outras opçoes:
    # CountVectorizer, TfidfVectorizer
    extractor = HashingVectorizer(analyzer=processor.process)
    return extractor.fit_transform([t for t, _ in documents])


text = '''
Olá, isso é um exemplo para o DataScience Meetup.
Alguém trouxe um guarda-chuva?
'''

if __name__ == '__main__':

    data = read_machado()
    print 'extracting features'
    features = extract_features(data).toarray()
    print 'extracting labels'
    labels = numpy.array([l for _, l in data])

    # Somente para garantir a distribuição das classes
    (train, test), _, _, _ = StratifiedKFold(labels, 4)

    print 'fit the classifier'
    clf = GaussianNB()
    clf.fit(features[train], labels[train])
    predictions = clf.predict(features[test])
    print classification_report(labels[test], predictions)
