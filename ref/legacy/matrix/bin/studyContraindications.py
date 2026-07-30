import pandas as pd
import re
from functools import cache

def wordsInText(text: str):
	text = re.sub(r'<.+?>', ' ', text)
	text = text.lower()
	text = re.sub(r'<', 'lt', text)
	text = re.sub(r'[^a-z0-9]+', ' ', text)
	text = re.sub(r'\s+', ' ', text)
	text = text.strip()
	return text.split(' ')


def indexContraindicationTexts(file: str):
	wordindex = {}
	with open(file, 'r') as cfile:
		for line in cfile:
			xml, text = line.strip().split('\t')
			words = wordsInText(text)
			if words[0] == 'contraindications': words = words[1:]
			if not words: continue
			xml = xml.lstrip('xmls/').rstrip('.xml.gz').lower()
			for word in words:
				if not word in wordindex: wordindex[word] = {}
				wordindex[word][xml] = 1
	return wordindex

#@cache
def textHits(words):
	xmls = {}
	for word in words:
		if not word in wordindex: continue
		for xml in wordindex[word]:
			if not xml in xmls: xmls[xml] = 1
			else: xmls[xml] += 1
	return xmls

def readExtractLog(file):
	info = {}
	with open(file, 'r') as f:
		for line in f:
			_, zip, _, _, xml, *_ = line.strip().split('\t')
			xml = xml.rstrip('.xml').lower()
			zip = zip.rstrip('.zip').lower()
			_, zip = zip.split('/')
			month, zip = zip.split('_')
			info[month[0:6]+'/'+xml] = zip
	return info

def readApprovalsList(file):
	apps = {}
	with open(file,'r') as f:
		for line in f:
			xml, app = line.strip().split('\t')
			xml = xml.lstrip('xmls/').rstrip('.xml.gz').lower()
			apps[xml] = app
	return apps

version = "1.4.1"
datafile = "data/contraindicationList-"+version+".xlsx"
dmcontrafile = "../DailyMed/extracted/contraindications.txt"
approvalsfile = "../DailyMed/extracted/approvals.txt"
DailyMedExtractLog = "../DailyMed/extract.log"
splset = readExtractLog(DailyMedExtractLog)

approvals = readApprovalsList(approvalsfile)
wordindex = indexContraindicationTexts(dmcontrafile)
#print('indexed', flush=True)
data = pd.read_excel(datafile, usecols=['active ingredient', 'contraindications', 'disease contraindicated', 'final normalized drug id', 'final normalized drug label', 'final normalized disease id', 'final normalized disease label'])
#print(data)
#print('read', flush=True)

#hist = [0] * 101
for index, row in data.iterrows():
	words = wordsInText(row['contraindications'])
	n = len(words)
	xmls = textHits(words)
	best = sorted(xmls.items(), key=lambda item: item[1], reverse=True)
	best = best[0:10]
	bestscore = 0
	keepxml = []
	keepset = []
	apps = []
	for xml in best:
		x = xml[0]
		score = xmls[x]/n
		if bestscore > 0 and score < bestscore: break
		bestscore = score
		keepxml.append(x)
		if x in splset:
			keepset.append(splset[x])
		if x in approvals:
			apps.append(approvals[x])
	
	print(row['final normalized drug id'], row['final normalized disease id'], "%.1f" % (100*bestscore), ','.join(keepxml), ','.join(keepset), ','.join(apps), sep='\t', flush=True)

#for i in range(101):
#	print(i, hist[i], sep='\t')