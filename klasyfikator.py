import nltk
import random
from nltk.corpus import movie_reviews
from string import punctuation
from nltk.corpus import stopwords
import pickle

dokumenty = []

for ocena_filmu in movie_reviews.categories():
    for fileid in movie_reviews.fileids(ocena_filmu):
        dokumenty.append((list(movie_reviews.words(fileid)), ocena_filmu))

#print(movie_reviews.categories())

random.shuffle(dokumenty)

wszystkie_wyrazy = []
for w in movie_reviews.words():
    wszystkie_wyrazy.append(w.lower())

#wszystkie_wyrazy = nltk.FreqDist(wszystkie_wyrazy)

wlasciwosci = list(wszystkie_wyrazy)

#print(wszystkie_wyrazy[:20])

"""normalizacja danych"""

stop_wyrazy = stopwords.words('english')

wszystkie_wyrazy = [w for w in wszystkie_wyrazy if w not in punctuation]
#print(wszystkie_wyrazy[:20])
wszystkie_wyrazy = [w for w in wszystkie_wyrazy if w not in stop_wyrazy]
#print(wszystkie_wyrazy[:20])

wszystkie_wyrazy = nltk.FreqDist(wszystkie_wyrazy)

#print(wszystkie_wyrazy.most_common(20))


wazne_wyrazy = wszystkie_wyrazy.most_common(3000)

lista = []

for w in wazne_wyrazy:
    lista.append(w[0])

wazne_wyrazy = lista

#print(wszystkie_wyrazy[:10])
#print(lista[:10])
""" koniec normalizacji"""

"""Funkcja która znajduje które spośród najważniejszych wyrazów znajdują się w danej recenzji"""
def zanjdz_cechy(dokument):
    wyrazy = set(dokument)
    cechy = {}
    for w in wazne_wyrazy:
        cechy[w] = (w in wyrazy)

    return cechy

#print((zanjdz_cechy(movie_reviews.words('neg/cv000_29416.txt'))))

cechy_dokumentow = []

"""Tworzymy dane do treningu i testowania, dane będą się
składać z krotek w których będzie się znajdować kategoria recenzji(pos lub neg)
oraz słownik z kluczami(najważniejszymi słowami) 
a wartość klucza będzie wynisiła true lub false"""
for (recenzja, kategoria) in dokumenty:
    cechy_dokumentow.append((zanjdz_cechy(recenzja), kategoria))

"""Spośród 2000 recenzji, do treningu użyjemy 1900 a do testów 100"""
dane_treningowe = cechy_dokumentow[:1900]
dane_testowe = cechy_dokumentow[1900:]

"""Klasyfikator oparty na algorytmie Bayesa. Model prawdopodobieństwa dla klasyfikatora jest modelem warunkowym."""
klasyfikator = nltk.NaiveBayesClassifier.train(dane_testowe)

print("Dokłdność klasyfikatora: ",(nltk.classify.accuracy(klasyfikator, dane_testowe))*100)

klasyfikator.show_most_informative_features(20)

""""""
print(klasyfikator.classify(dane_testowe[0][0]) , " : " , dane_testowe[0][1])
print(klasyfikator.classify(dane_testowe[1][0]) , " : " , dane_testowe[1][1])
print(klasyfikator.classify(dane_testowe[2][0]) , " : " , dane_testowe[2][1])

"""Aby zapisać nasz klasyfikator można użyć modułu pickle"""
#zapisz_klasyfikator = open("klasyfikator_bayes.pickle","wb")
#pickle.dump(klasyfikator, zapisz_klasyfikator)
#zapisz_klasyfikator.close()

"""Aby odczytać zapisany klasyfikator"""
#klasyfikator_f = open("klasyfikator_bayes.pickle", "rb")
#klasyfikator = pickle.load(classifier_f)
#klasyfikator_f.close()
