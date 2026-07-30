import gzip
import json
import sys

def readNodes(file):
	names = {}
	cats = {}
	with gzip.open(file + '.jsonl.gz', 'r') as f:
		for line in f:
			try:
				jobj = json.loads(line.strip())
			except json.JSONDecodeError as e:
				print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
				continue
			id = jobj['id']
			names[id] = jobj['name']
			cat = jobj['category']
			if isinstance(cat, list):
				print('list', cat)
				cats[id] = cat[0]
			else:
				print('str', cat)
				cats[id] = cat
	return names, cats

def processJsonlines(file, headers):
	with gzip.open(file + '.jsonl.gz', 'r') as f, gzip.open(file + '.tsv.gz', 'wt') as of:
		print(*headers, sep='\t', file=of)
		for line in f:
			try:
				jobj = json.loads(line.strip())
			except json.JSONDecodeError as e:
				print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
				continue
			row = []
			for h in headers:
				if h in jobj:
					if h == 'category':
						cat = jobj[h]
						if isinstance(cat, list):
							row.append(cat[0])
						else:
							row.append(cat)
					else:
						row.append(jobj[h])
				elif h+'s' in jobj:
					row.append(','.join(jobj[h+'s']))
				elif h == 'supporting_spls' and 'has_evidence' in jobj:
					row.append(','.join(jobj['has_evidence']))
				else:
					row.append('NA')
			print(*row, sep='\t', file=of)



## MAIN
# General definitions
base = sys.argv[1]
version = sys.argv[2]

nodesfile = base+'_nodes_v'+version
edgesfile = base+'_edges_v'+version

nodesFileHeaders = '''id name category'''.split()
edgesFileHeaders = '''id subject	predicate	object	subject_name	object_name object_modifier knowledge_level	agent_type	 approval N_cases supporting_spls'''.split()


#names, categories = readNodes(nodesfile)
#print(len(names))
processJsonlines(nodesfile, nodesFileHeaders)
processJsonlines(edgesfile, edgesFileHeaders)
