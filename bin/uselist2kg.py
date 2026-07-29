#import pandas as pd
from sqlite_utils import Database
from textProcessFunctions import *
from functools import cache
import gzip
import re

def listCategories():
	cats = {}
	for row in babel.execute("select * from categories"):
		cats[row[0]] = row[1]
	return cats

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

@cache
def interpretTerm(term, categories):
	#doLog("interpreting-term", term)
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

def curieIdInfo(curieid):
	for row in babel.execute("select * from names where id = ?", (curieid,)):
		curie = row[1]
		curieInfo[curie] = [categories[row[2]], row[3], row[4]]
		return curie, *curieInfo[curie] #row[1], categories[row[2]], row[3], row[4]

def getCurieInfo(curie):
	if not curie in curieInfo: 
		for row in babel.execute("select id from synonyms where l1 = ?", (curie.lower(),)):
			curieid = row[0]
			curieIdInfo(curieid)
	return curieInfo[curie]

def split_prefix_number(s):
	# This regex looks for a sequence of non-digit characters (the prefix)
	# followed by a sequence of digit characters (the number).
	# The parentheses create capturing groups for easy extraction.
	match = re.match(r'([^\d]+)(\d+)', s)
	if match:
		prefix = match.group(1)
		number = match.group(2)
		return prefix, number
	else:
		return None, None # Or raise an error, depending on desired behavior

def readApprovals(file):
	info = {}
	prefix = {}
	ambiguous = {}
	with open(file, 'r') as apfile:
		for line in apfile:
			xml, code = line.strip().split('\t')
			pref, nda = split_prefix_number(code)
			if not nda in prefix: prefix[nda] = {pref: 1}
			elif pref in prefix[nda]: pass
			else:
				doLog('ambiguous approval code', nda, pref, list(prefix[nda]))
				prefix[nda][pref] = 1
				ambiguous[nda] = 1
			if not nda in info: info[nda] = []
			info[nda].append(xml)
	return info, prefix, ambiguous

def readIngredients(file):
	info = {}
	uniis = {}
	xmls = {}
	with open(file, "r") as ingfile:
		for line in ingfile:
			xml, unii, name = line.strip().split('\t')
			name = name.strip()
			if not xml in info: info[xml] = {}
			info[xml][name] = 1
			uniis[name] = unii if unii else 'NA'
			if not unii in xmls: xmls[unii] = {}
			xmls[unii][xml] = 1
	return info, uniis, xmls

def readIndications(file):
	info = {}
	#qual = {}
	with gzip.open(file, "rt") as indfile:
		for line in indfile:
			if line[0] == '#': continue
			_, _, term, curie = line.strip().split('\t') ### could have a 'qualifier' field at the end
			#next if $qualifier eq '###';
			term = term.strip('"')
			#print(term, curie, flush=True)
			info[term] = curie
			#qual[term] = qualifier if $qualifier;
	return info

def readDailyMedSupport(file):
	info = {}
	with open(file, 'r') as dmsupfile:
		for line in dmsupfile:
			xml, curie, pos, term = line.strip().split('\t')
			if not curie in info: info[curie] = {}
			info[curie][xml] = pos
	return info

def saveNode(saved, file, subject, subj_name, category):
	if subject in saved: return
	print(subject, subj_name, category, file=file, sep='\t', flush=True)
	saved[subject] = 1

def saveEdge(saved, file, subj, pred, obj, *fields):
	key = '\t'.join([str(subj), pred, obj])
	if key in saved: return
	print(subj, pred, obj, *fields, file=file, sep='\t', flush=True)
	saved[key] = 1

def doLog(*strings):
	print(*strings, sep='\t', file=wfile, flush=True)

## MAIN
# General definitions
outbase = "results/uses2indi"
DMdir = "DailyMed"
FAERSdir = "FAERS"
babelFile = '/ssd2/sqlite/BABEL.db'
nodesFileHeaders = ['id', 'name', 'category']
edgesFileHeaders = '''subject	predicate	object	subject_name	object_name object_modifier knowledge_level	agent_type	 approval N_cases supporting_spls'''.split()
FAERSuses = FAERSdir+'/results/uses.txt.gz'
interventionCategories = tuple(['ChemicalEntity', 'SmallMolecule', 'Drug', 'MolecularMixture', 'ComplexMolecularMixture', 'ChemicalMixture', 'Protein'])
conditionCategories = tuple(['Disease', 'PhenotypicFeature'])


# Preparation
babel = Database(babelFile)
curieInfo = {}
savedNodes = {}
savedEdges = {}
categories = listCategories()
wfile = open(outbase+'-warnings.txt', 'w')

print("reading approvals", flush=True)
nda2xml, prefix, ambiguous = readApprovals(DMdir+'/extracted/approvals.txt')
print("reading ingredients", flush=True)
xml2ing, uniis, unii2xml = readIngredients(DMdir+'/results/singleton_active_ingredients.txt')
print("reading indications", flush=True)
term2curie = readIndications(FAERSdir+'/results/FAERS-indication-terms.txt.gz')
print("reading terms in indications", flush=True)
supportInDailyMed = readDailyMedSupport(DMdir+'/results/terms-in-indications.txt')

nfile = gzip.open(outbase+'-nodes.txt.gz', 'wt')
efile = gzip.open(outbase+'-edges.txt.gz', 'wt')
print('\t'.join(nodesFileHeaders), file=nfile)
print('\t'.join(edgesFileHeaders), file=efile)

print("processing indications", flush=True)
seen = {}
with gzip.open(FAERSuses, 'rt') as usesfile:
	for line in usesfile:
		n = 0
		ndas = ''
		ingredient = ''
		indication = ''
		try:
			cases, ndas, ingredient, indication = line.strip().split('\t')
		except:
			try:
				cases, ndas, ingredient = line.strip().split('\t')
			except:
				print(line)
				continue
		if cases == 'CASES': continue
		objects = interpretConditions('', indication)
		if not objects:
			if indication in seen: continue
			seen[indication] = 1
			doLog('cannot interpret condition', indication)
			continue
		try: curie = term2curie[indication]
		except:
			if indication in seen: continue
			seen[indication] = 1
			doLog('no curie for', indication)
			continue
		if not curie in supportInDailyMed: ##### we need this just to avoid weird mismappings... but this loses applied_to_treat content
			if curie in seen: continue
			seen[curie] = 1
			doLog('nothing in DailyMed for', curie)
			continue
		try: cat, name, taxon = getCurieInfo(curie)
		except: 
			if curie in seen: continue
			seen[curie] = 1
			doLog('no info for', curie)
			continue
		if not re.search(r'\w', name):
			doLog('nothing in name', curie, name)
			continue
		### my $q = $qualifiers->{$indication};
		#if nda in ambiguous:
		#	if nda in seen: continue
		#	seen[nda] = 1
		#	doLog('skipping ambiguous', nda)
		#	continue
		for nda in ndas.split(','):
			ingredients = {}
			if not nda in nda2xml:
				if nda in seen: continue
				seen[nda] = 1
				doLog('missing NDA', nda)
				continue
			for xml in nda2xml[nda]:
				if not xml in xml2ing: continue
				for ing in xml2ing[xml]:
					if not ing in ingredients: ingredients[ing] = 0
					ingredients[ing] += 1
			inglist = sorted(ingredients.items(), key=lambda item: item[1], reverse=True)
			display_nda = []
			for pref in sorted(prefix[nda]):
				display_nda.append(pref + nda)
			display_nda = ','.join(display_nda)
			for ing in inglist:
				if not re.search(ing[0], ingredient): continue
				unii = uniis[ing[0]]
				if not unii in unii2xml: continue

				saveNode(savedNodes, nfile, unii, ing[0], 'biolink:'+'ChemicalEntity')
				saveNode(savedNodes, nfile, curie, name, 'biolink:'+cat)

				supportingSPLs = {}
				for xml in unii2xml[unii]:
					if not curie in supportInDailyMed or not xml in supportInDailyMed[curie]: continue
					xml = xml.lstrip('xmls/')
					xml = xml.rstrip('.xml.gz')
					supportingSPLs[xml] = 1
				SPLlist = ','.join(list(supportingSPLs))

				predicate = 'biolink:applied_to_treat'
				knowledge_level = 'observation'
				agent_type = 'text_mining_agent'
				saveEdge(savedEdges, efile, 
					unii, predicate, curie,
					ing[0], name,
					'(qual)',
					knowledge_level, agent_type,
					'NA', cases, 'NA'
					)

				if SPLlist:
					predicate = 'biolink:treats'
					knowledge_level = 'knowledge_assertion'
					saveEdge(savedEdges, efile, 
						unii, predicate, curie,
						ing[0], name,
						'(qual)',
						knowledge_level, agent_type,
						display_nda, 0, SPLlist
						)
