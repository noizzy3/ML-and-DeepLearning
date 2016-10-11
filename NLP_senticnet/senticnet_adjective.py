import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import numpy as np
from senti_classifier import senti_classifier
from senti_classifier.senti_classifier import synsets_scores
import difflib
sm = difflib.SequenceMatcher(None)
#print synsets_scores[syn.name()]['pos']
target=open("subham_sekhar_senticnet.txt",'w')
def check(a,b):
	if(len(a) > 2 and len(b)>2):
		if a[0] == b[0] and a[1] == b[1] and a[2] == b[2] :
			return 0
	return 1


positive = ["#joyful" , "#interesting" , "#surprising" , "#admirable"]
negative = ["#sad", "#scared" , "#angry" , "#disgusting"]

#change below manually

positive_form = ['joyful','interesting','surprising','admirable']
negative_form = ['unhappy','scared','inflame','disgust']

target.write("senticnet = {}\n")
for syn in list(wn.all_synsets('a')):
	s = str(syn.lemmas()[0].name())
	sm.set_seq2(s)
	synonyms = []
	score = []
	if synsets_scores[syn.name()]['pos'] == 0.0 and synsets_scores[syn.name()]['neg'] == 0.0:
		continue

	if synsets_scores[syn.name()]['pos'] >= synsets_scores[syn.name()]['neg'] :
		for x in positive_form :
			sm.set_seq1(x)
			score.append(sm.ratio())
		synonyms.append(positive[np.argmax(score)])
		score[np.argmax(score)] = -10
		synonyms.append(positive[np.argmax(score)])
	else :
		for x in negative_form :
			sm.set_seq1(x)
			score.append(sm.ratio())
		synonyms.append(negative[np.argmax(score)])
		score[np.argmax(score)] = -10
		synonyms.append(negative[np.argmax(score)])

	for l in syn.lemmas() :
		w=nltk.stem.WordNetLemmatizer().lemmatize(str(l.name())) #lemmatize the word
		if check(s,w) :
			synonyms.append(w) #str converts it from unicode to string
		if len(synonyms) > 6 :
			break
	if len(synonyms) == 7 :
		target.write(s)
		target.write(" = [\"")
		count=0
		for x in synonyms:
			if count :
				target.write(",\"")
			count+=1
			target.write(x)
			target.write("\"")
		target.write("]")
		target.write("\n")
target.close()