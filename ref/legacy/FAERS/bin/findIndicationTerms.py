from sqlite_utils import Database
import pandas as pd
import functools
import gzip
from textProcessFunctions import *

def listCategories():
	cats = {}
	for row in babel.execute("select * from categories"):
		cats[row[0]] = row[1]
	return cats

def curieIdInfo(curieid):
	for row in babel.execute("select * from names where id = ?", (curieid,)):
		curie = row[1]
		curieInfo[curie] = [categories[row[2]], row[3], row[4]]
		return curie, *curieInfo[curie] #row[1], categories[row[2]], row[3], row[4]

def BABELmatches(term: str, **kwargs):
	if len(term)<2 or term == 'nan': return []
	max_matches = kwargs.get('max_matches', 100)
	matches = []
	mterm = level1(term)
	if not mterm: return matches
	for row in babel.execute("select * from synonyms where L1 = ?", (mterm,)):
		matches.append(row[0])
		if len(matches) >= max_matches: return matches
	if not matches:
		mterm = level3acc(term)
		if not mterm: return matches
		for row in babel.execute("select * from synonyms where L3 = ?", (mterm,)):
			matches.append(row[0])
	return matches

@functools.cache
def interpretTerm(term, categories):
	#print("#interpreting-term", term, file=wfile, sep='\t', flush=True)
	curies = {}
	matches = BABELmatches(clean_string(term))
	for curieid in matches:
		#pref = preferredCurie(curie) ### won't be needed
		curie, cat, _, _ = curieIdInfo(curieid)
		if categories and not cat in categories: continue
		curies[curie] = 1
	return curies

def interpretTerms(nctid, terms, categories):
	results = {}
	for term in terms:
		if term == 'nan' or term == 'NA' or term == 'Na': continue
		curies = interpretTerm(term, categories)
		if len(curies) == 0: continue
		for curie in curies:
			if not curie in results: results[curie] = {}
			results[curie][term] = 1
			print('#interpreting', nctid, term, curie, file=wfile, sep='\t')
	return results

def interpretInterventions(nctid, text):
	if text == 'nan': return []
	parts = re.split(r'\s*\|\s*|\s*,\s*|\s+or\s+|\s+and\s+|\s+with\s+|\s+plus\s+|\s+combined\s+|\s*\:\s*|\s*\+\s*|\s*\(\s*|\s*\)\s*|\s*\/\s*|\s*;\s*', text)
	return interpretTerms(nctid, parts, interventionCategories)

def interpretConditions(nctid, text):
	if text == 'nan': return
	parts = text.split('|')
	return interpretTerms(nctid, parts[::2], conditionCategories)








## MAIN
# General definitions
indications_file = "results/indications.txt"
outfile = "results/FAERS-indication-terms.txt"
babelFile = '/ssd2/sqlite/BABEL.db'
interventionCategories = tuple(['ChemicalEntity', 'SmallMolecule', 'Drug', 'MolecularMixture', 'ComplexMolecularMixture', 'ChemicalMixture'])
conditionCategories = tuple(['Disease', 'PhenotypicFeature'])

# Preparation
babel = Database(babelFile)
curieInfo = {}
categories = listCategories()
wfile = open('warnings.txt', 'w')

indications = pd.read_csv(indications_file, sep='\t', header=None, names=['count', 'indication'])
total = indications['count'].sum()

cumulative = 0
with gzip.open('results/FAERS-indication-terms.txt.gz', 'wt') as ofile:
	for index, row in indications.iterrows():
		count = row['count']
		cumulative = cumulative+count
		indication = row['indication']
		conditions_dict = interpretConditions('', indication)
		conditions_list = list(conditions_dict)
		try: condition = conditions_list[0]
		except: condition = '???'
		print(count, int(cumulative/total*10000)/100, indication, condition, sep='\t', file=ofile)
