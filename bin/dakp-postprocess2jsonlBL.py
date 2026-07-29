import gzip
import json
import pandas as pd # type: ignore
import csv
#from numbers import Number
from sqlite_utils import Database # type: ignore
from textProcessFunctions import *
from functools import cache

#from labeler.labeler import namespace_uuid # type: ignore
from uuid import UUID, uuid3

@cache
def basespace(domain: str) -> UUID:
	namespace: UUID = UUID("00000000-0000-0000-0000-000000000000")
	return uuid3(namespace, domain)

def namespace_uuid(domain: any, *values: list[any]) -> str:
	domain = str(domain)
	values = [str(x) for x in values if x]
	domainspace: UUID = basespace(domain)
	joined: str = "\t".join(values)
	return str(uuid3(domainspace, joined))


def isNaN(num):
	return num != num

def doLog(*strings):
	print(*strings, sep='\t', file=wfile, flush=True)

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



def isNaN(num):
	return num != num

def determineCurrentVersion(file):
	with open(file, 'r') as f:
		version = f.readline()
	return version

def readNodes(file):
	nodes = {}
	df = pd.read_csv(file, compression='gzip', sep='\t', quotechar='"')
	for _, row in df.iterrows():
		id = row['id']
		if id in nodes:
			print("ignoring duplicate node", id)
			continue
		nodes[id] = row.to_dict()
		nodes[id]['category'] = [nodes[id]['category']]
	return nodes

def readEdges(file, seen, edges, nodes, approved):
	df = pd.read_csv(file, compression='gzip', sep='\t', quotechar='"', quoting=csv.QUOTE_NONE, on_bad_lines='warn')
	for _, row in df.iterrows():
		subject = row['subject']
		predicate = row['predicate']
		object = row['object']

		# normalize subject and object, build nodes
		subjs = list(interpretTerm(subject, interventionCategories))
		if len(subjs) == 0:
			subjs = list(interpretTerm(row['subject_name'], interventionCategories))
		if len(subjs) == 0:
			if subject in seen: continue
			seen[subject] = 1
			doLog('unknown curie', subject)
			continue
		subj = subjs[0]
		if not subj in nodes:
			subj_cat, subj_name, _ = getCurieInfo(subj)
			nodes[subj] = {'id': subj, 'name': subj_name, 'category': subj_cat}

		objs = list(interpretTerm(object, conditionCategories))
		if len(objs) == 0:
			if object in seen: continue
			seen[object] = 1
			doLog('unknown curie', object)
			continue
		obj = objs[0]
		if not obj in nodes:
			obj_cat, obj_name, _ = getCurieInfo(obj)
			nodes[obj] = {'id': obj, 'name': obj_name, 'category': obj_cat}
			#print(subject, subj, subjcat, subjname, sep='\t', flush=True)

		# process objects to generate triples
		key = namespace_uuid('drug_approvals_kp', subj, predicate, obj)
		if key in edges:
			### previously observed triple, update / add new content
			edges[key]['N_cases'] = edges[key]['N_cases'] + row['N_cases']
			if not isNaN(row['approval']):
				if not 'approvals' in edges[key]:
					edges[key]['approvals'] = {}
				for nda in row['approval'].split(','):
					edges[key]['approvals'][nda] = 1
				for spl in row['supporting_spls'].split(','):
					edges[key]['has_evidence'][spl] = 1
			continue
		else:
			# first time we see this triple, set edge up
			if obj_cat == 'biolink:Disease': edge_cat = 'biolink:EntityToDiseaseAssociation' #'biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation'
			elif obj_cat == 'biolink:PhenotypicFeature': edge_cat = 'biolink:EntityToPhenotypicFeatureAssociation' #'biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation'
			else: edge_cat = 'biolink:EntityTo'+obj_cat+'Association'
			edges[key] = {
				'id': key,
				'subject': subj,
				'predicate': predicate,
				'object': obj,
				'subject_name': nodes[subj]['name'],
				'object_name': nodes[obj]['name'],
				'category': [edge_cat],
				'knowledge_level': row['knowledge_level'],
				'agent_type': dakp_at, #row['agent_type'],
				'N_cases': row['N_cases'] ### not biolink, use dataset_count association slot?
			}
			if predicate == 'biolink:treats':
				if not subj in approved: approved[subj] = {}
				approved[subj][obj] = 1
				#edges[key]['clinical_approval_status'] = "approved_for_condition" ### need to be more precise here: approved_for_condition vs. fda_approved_for_condition vs. post_approval_withdrawal
				edges[key]['sources'] = [
					{
						'resource_id': "infores:multiomics-drugapprovals",
						'resource_role': "primary_knowledge_source",
						'upstream_resource_ids': [ "infores:dailymed", "infores:faers" ],
						'source_record_urls': [ "https://db.systemsbiology.net/gestalt/cgi-pub/KGinfo.pl?id="+key ]
					},
					{
						'resource_id': "infores:faers",
						'resource_role': "supporting_data_source",
					},
					{
						'resource_id': "infores:dailymed",
						'resource_role': "supporting_data_source",
						### could add here source_record_urls to point at the SPLs themselves, but - bloat!
					},
				]
			elif predicate == 'biolink:applied_to_treat':
				#edges[key]['N_cases'] = row['N_cases']
				#edges[key]['clinical_approval_status'] = "off_label_use" ### or not_approved_for_condition ? off_label implies approved for something else, not_approved implies no approval at all
				edges[key]['sources'] = [
					{
						'resource_id': "infores:multiomics-drugapprovals",
						'resource_role': "aggregator_knowledge_source",
						'upstream_resource_ids': [ "infores:dailymed", "infores:faers" ],
						'source_record_urls': [ "https://db.systemsbiology.net/gestalt/cgi-pub/KGinfo.pl?id="+key ]
					},
					{
						'resource_id': "infores:faers",
						'resource_role': "primary_knowledge_source",
					},
					{
						'resource_id': "infores:dailymed",
						'resource_role': "supporting_data_source",
					},
				]
			else:
				doLog('unexpected predicate', subj, predicate, obj)
				continue
			if not isNaN(row['approval']):
				### use has_evidence and Publication
				if not 'approvals' in edges[key]:
					edges[key]['approvals'] = {}
				for nda in row['approval'].split(','):
					edges[key]['approvals'][nda] = 1
				edges[key]['has_evidence'] = {}
				for spl in row['supporting_spls'].split(','):
					edges[key]['has_evidence'][spl] = 1

	#return edges

def addContraindications(file, seen, edges, nodes):
	df = pd.read_csv(file, compression='gzip', sep='\t', quotechar='"', quoting=csv.QUOTE_NONE, on_bad_lines='warn')
	for _, row in df.iterrows():
		#print(row, flush=True)
		subject = row['subject']
		predicate = contra_pred #row['predicate']
		object = row['object']

		# normalize subject and object, build nodes
		subjs = list(interpretTerm(subject, interventionCategories))
		if len(subjs) == 0:
			subjs = list(interpretTerm(row['subject_name'], interventionCategories))
		if len(subjs) == 0:
			if subject in seen: continue
			seen[subject] = 1
			doLog('unknown curie', subject)
			continue
		subj = subjs[0]
		if not subj in nodes:
			subj_cat, subj_name, _ = getCurieInfo(subj)
			nodes[subj] = {'id': subj, 'name': subj_name, 'category': subj_cat}
		else:
			subj_cat = nodes[subj]['category']
			subj_name = nodes[subj]['name']

		objs = list(interpretTerm(object, conditionCategories))
		if len(objs) == 0:
			if object in seen: continue
			seen[object] = 1
			doLog('unknown curie', object)
			continue
		obj = objs[0]
		if not obj in nodes:
			obj_cat, obj_name, _ = getCurieInfo(obj)
			nodes[obj] = {'id': obj, 'name': obj_name, 'category': obj_cat}
			#print(subject, subj, subjcat, subjname, sep='\t', flush=True)
		else:
			obj_cat = nodes[obj]['category']
			obj_name = nodes[obj]['name']

		# process objects to generate triples
		key = namespace_uuid('drug_approvals_kp', subj, predicate, obj)
		if key in edges:
			### previously observed triple, update / add new content
			edges[key]['N_cases'] = edges[key]['N_cases'] + row['N_cases']
			if not isNaN(row['approval']):
				if not 'approvals' in edges[key]:
					edges[key]['approvals'] = {}
				for nda in row['approval'].split(','):
					edges[key]['approvals'][nda] = 1
				for spl in row['supporting_spls'].split(','):
					if not 'has_evidence' in edges[key]: edges[key]['has_evidence'] = {}
					edges[key]['has_evidence'][spl] = 1
			continue
		else:
			# first time we see this triple, set edge up
			if obj_cat == 'biolink:Disease': edge_cat = 'biolink:EntityToDiseaseAssociation' #'biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation'
			elif obj_cat == 'biolink:PhenotypicFeature': edge_cat = 'biolink:EntityToPhenotypicFeatureAssociation' #'biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation'
			else: edge_cat = 'biolink:EntityTo'+obj_cat+'Association'
			edges[key] = {
				'id': key,
				'subject': subj,
				'predicate': predicate,
				'object': obj,
				'subject_name': nodes[subj]['name'],
				'object_name': nodes[obj]['name'],
				'category': [edge_cat],
				'knowledge_level': contra_kl, #row['knowledge_level'],
				'agent_type': contra_at, #row['agent_type'],
				#'N_cases': row['N_cases'] ### not biolink, use dataset_count association slot?
			}
			if predicate == contra_pred:
				#edges[key]['clinical_approval_status'] = "?"
				edges[key]['sources'] = [
					{
						'resource_id': "infores:multiomics-drugapprovals",
						'resource_role': "aggregator_knowledge_source",
						'upstream_resource_ids': [ "infores:dailymed", "infores:medi" ],
						'source_record_urls': [ "https://db.systemsbiology.net/gestalt/cgi-pub/KGinfo.pl?id="+key ]
					},
					{
						'resource_id': "infores:medi",
						'resource_role': "primary_knowledge_source",
						'upstream_resource_ids': [ "infores:dailymed"],
					},
					{
						'resource_id': "infores:dailymed",
						'resource_role': "supporting_data_source",
						### could add here source_record_urls to point at the SPLs themselves, but - bloat!
					},
				]
			else:
				doLog('unexpected predicate', subj, predicate, obj)
				continue
			if not isNaN(row['approval']):
				### use has_evidence and Publication
				if not 'approvals' in edges[key]:
					edges[key]['approvals'] = {}
				for nda in row['approval'].split(','):
					edges[key]['approvals'][nda] = 1
				edges[key]['has_evidence'] = {}
				if not isNaN(row['supporting_spls']):
					for spl in row['supporting_spls'].split(','):
						edges[key]['has_evidence'][spl] = 1

	return edges


def postProcessNodes(nodes):
	#			nodes[obj] = {'id': obj, 'name': obj_name, 'category': obj_cat}
	for node in nodes:
		nodes[node]['category'] = ['biolink:'+nodes[node]['category']]

def postProcessEdges(edges, approved):
	seen = {}
	for edge in edges:
		subj = edges[edge]['subject']
		pred = edges[edge]['predicate']
		obj = edges[edge]['object']
		if pred == 'biolink:treats' and 'N_cases' in edges[edge]:
			del edges[edge]['N_cases']
		if subj in approved and obj in approved[subj]:
			edges[edge]['clinical_approval_status'] = "approved_for_condition"
		elif pred == 'biolink:applied_to_treat':
			edges[edge]['clinical_approval_status'] = "off_label_use" ## or not_approved_for_condition ? off_label implies approved for something else, not_approved implies no approval at all
		else:
			pass
			#edges[edge]['clinical_approval_status'] = "off_label_use" ## or not_approved_for_condition ? off_label implies approved for something else, not_approved implies no approval at all
		if 'approvals' in edges[edge]:
			edges[edge]['approvals'] = sorted(list(edges[edge]['approvals'])) ### temporary, need to fold into has_evidence
		if 'has_evidence' in edges[edge]:
			spls = []
			for spl in edges[edge]['has_evidence']:
				try:
					spls.append(spl2set[spl])
				except:
					if spl in seen: continue
					seen[spl] = 1
					doLog('no info on', spl)
			edges[edge]['has_evidence'] = sorted(spls) ### transform into actual linkable uuids

def saveJsonLines(outfile, content):
	with gzip.open(outfile + '.jsonl.gz', 'wt') as outf:
		for obj in content:
			print(json.dumps(content[obj]), file=outf)

def readDailyMedSets(file):
	info = {}
	with open(file, 'r') as dmes:
		for line in dmes:
			xml, splset = line.rstrip().split('\t')
			xml = xml[5:-7] # delete xmls/ and .xml.gz
			info[xml.lower()] = 'dailymed:'+splset.lower()
	return info

## MAIN
# General definitions
nodesFile = "results/drug2indi-nodes.txt.gz"
edgesFile = "results/drug2indi-edges.txt.gz"
outbase   = "results/drug_approvals_kg_"
DailyMedSets = "DailyMed/extracted/sets.txt"
babelFile = '/ssd2/sqlite/BABEL.db'
interventionCategories = tuple(['ChemicalEntity', 'SmallMolecule', 'Drug', 'MolecularMixture', 'ComplexMolecularMixture', 'ChemicalMixture', 'Protein'])
conditionCategories = tuple(['Disease', 'PhenotypicFeature'])
dakp_at = 'manual_validation_of_automated_agent'

# Matrix contraindications
matrix_version = "1.4.1"
contra_edges_file = "matrix/results/contraindications_kg_edges.tsv.gz"
#supporting_spls_file = "matrix/results/xmlspls_supporting_contraindications.tsv"
contra_pred = 'biolink:contraindicated_in'
contra_kl = 'knowledge_assertion'
contra_at = 'manual_validation_of_automated_agent'


# Preparation
babel = Database(babelFile)
curieInfo = {}
categories = listCategories()
spl2set = readDailyMedSets(DailyMedSets)
#for key in spl2set:
#	print(key, spl2set[key], sep='\t', flush=True)
#exit()
version = "0.5.5" #determineCurrentVersion("version.latest")
### increment version if required, then save it
outfileEdges = outbase + "edges_v" + version
outfileNodes = outbase + "nodes_v" + version
wfile = open(outbase+'-warnings.txt', 'w')
seen = {}
edges = {}
nodes = {}
approved = {}


print("reading DAKP edges", flush=True)
readEdges(edgesFile, seen, edges, nodes, approved)

print("reading MEDI contraindications", flush=True)
addContraindications(contra_edges_file, seen, edges, nodes)

#print("adding treats edges", flush=True)
#addTreatsEdges(edges)

print("post-processing", flush=True)
postProcessNodes(nodes)
postProcessEdges(edges, approved)

print("saving", flush=True)
saveJsonLines(outfileNodes, nodes)
saveJsonLines(outfileEdges, edges)
