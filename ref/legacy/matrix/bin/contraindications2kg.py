import pandas as pd
import uuid
from functools import cache
from sqlite_utils import Database
import functools
from textProcessFunctions import *
import gzip

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

def saveNode(saved, file, subject, subj_name, category):
	if subject in saved: return
	print(subject, subj_name, category, file=file, sep='\t', flush=True)
	saved[subject] = 1

def saveEdge(saved, file, id, subj, pred, obj, *fields):
	key = '\t'.join([str(subj), obj])
	if key in saved: return
	print(id, subj, pred, obj, *fields, file=file, sep='\t', flush=True)
	saved[key] = 1

## MAIN
# General definitions
version = "1.4.1"
datafile = "data/contraindicationList-"+version+".xlsx"
supporting_spls_file = "results/xmlspls_supporting_contraindications.tsv"
pred = 'biolink:contraindicated_in'
kl = 'knowledge_assertion'
at = 'text_mining_assisted'
babelFile = '/ssd2/sqlite/BABEL.db'
outbase = 'results/contraindications_kg'
interventionCategories = tuple(['ChemicalEntity', 'SmallMolecule', 'Drug', 'MolecularMixture', 'ComplexMolecularMixture', 'ChemicalMixture'])
conditionCategories = tuple(['Disease', 'PhenotypicFeature'])

# Preparation
babel = Database(babelFile)
categories = listCategories()
basespace = uuid.uuid5(uuid.NAMESPACE_URL, "00000000-0000-0000-0000-000000000000")
namespace = uuid.uuid5(basespace, 'drug_approvals_kp')
nodesFileHeaders = ['id', 'name', 'category']
edgesFileHeaders = '''id	subject	predicate	object	subject_name	object_name	object_modifier	knowledge_level	agent_type	approval	N_cases	supporting_spls	unii'''.split()

#spls = pd.read_csv(supporting_spls_file, sep='\t', names=['intervention', 'condition', 'score', 'codes'])
curieInfo = {}
savedNodes = {}
savedEdges = {}
xmls = {}
codes = {}
approvals = {}
with open(supporting_spls_file, 'r') as supp:
	for line in supp:
		try: intervention, condition, score, xml, code, apps = line.strip().split('\t')
		except: continue
		xmls[intervention+'$'+condition] = xml
		codes[intervention+'$'+condition] = code
		approvals[intervention+'$'+condition] = apps

data = pd.read_excel(datafile, usecols=['active ingredient', 'contraindications', 'disease contraindicated', 'final normalized drug id', 'final normalized drug label', 'final normalized disease id', 'final normalized disease label'])

with gzip.open(outbase+'_nodes.tsv.gz', 'wt') as nfile, gzip.open(outbase+'_edges.tsv.gz', 'wt') as efile:
	print('\t'.join(nodesFileHeaders), file=nfile)
	print('\t'.join(edgesFileHeaders), file=efile)

	for index, row in data.iterrows():
		subj = row['final normalized drug id']
		interpretTerm(subj, interventionCategories)
		try: subj_cat, subj_name, taxon = curieInfo[subj]
		except:
			print('cannot interpret', subj, sep='\t', flush=True)
			continue
		obj = row['final normalized disease id']
		interpretTerm(obj, conditionCategories)
		try: obj_cat, obj_name, taxon = curieInfo[obj]
		except:
			print('cannot interpret', obj, sep='\t', flush=True)
			continue
		#subjname = row['final normalized drug label']
		#objname = row['final normalized disease label']
		saveNode(savedNodes, nfile, subj, subj_name, 'biolink:'+subj_cat)
		saveNode(savedNodes, nfile, obj, obj_name, 'biolink:'+obj_cat)
		#support = spls.loc[(spls['intervention'] == subj) & (spls['condition'] == obj), 'codes']
		try: xml = xmls[subj+'$'+obj]
		except: xml = ''
		try: code = codes[subj+'$'+obj]
		except: code = ''
		try: app = approvals[subj+'$'+obj]
		except: app = ''
		id = uuid.uuid5(namespace, '\t'.join([subj, pred, obj]))
		saveEdge(savedEdges, efile, 
			str(id), subj, pred, obj, 
			subj_name, obj_name, 'NA',
			kl, at, 
			app, 'NA', xml, #code, 
			subj if subj[0:4] == 'UNII' else 'NA', 
			)
