from os import walk
from os import mkdir
import sys
import gzip
from lxml import etree
from ordered_set import OrderedSet # type: ignore

# <document
#   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
#   xmlns="urn:hl7-org:v3"
#   xsi:schemaLocation="urn:hl7-org:v3 http://www.accessdata.fda.gov/spl/schema/spl.xsd">

hl7v3 = "urn:hl7-org:v3"
namespaces = {"v": hl7v3}
codetag = '{'+hl7v3+'}code'
nametag = '{'+hl7v3+'}name'

def parse_set(data):
    set_codes = list()
    for child in data:
        if child.tag == '{'+hl7v3+'}setId':
            return child.get('root').lower()
    return ''

def parse_approvals(data):
    app_codes = list()
    codes = list()
    try:
        found = data.xpath("//v:subjectOf/v:approval/v:id[@root='2.16.840.1.113883.3.150']",
                           namespaces=namespaces)
        for item in found:
            app_codes.append(item.attrib['extension'])
        found = data.xpath("//v:subjectOf/v:approval/v:code[@codeSystem='2.16.840.1.113883.3.26.1.1']",
                           namespaces=namespaces)
        for item in found:
            codes.append(item.attrib['code'])
    except: pass
    return app_codes, codes

def parse_ingredients(data, active):
    ingredients = {}
    try:
        if active == 'active':
            found = data.xpath("//v:activeMoiety/v:activeMoiety"
                               "| //v:activeIngredientSubstance",
                               namespaces=namespaces)
        else:
            found = data.xpath("//v:inactiveIngredient/v:inactiveIngredientSubstance"
                               "| //v:ingredient[@classCode='IACT']/v:ingredientSubstance",
                               namespaces=namespaces)
        for item in found:
            unii = 'NA'
            for subitem in item:
                if subitem.tag == codetag:
                    unii = 'UNII:'+subitem.attrib['code']
                elif subitem.tag == nametag:
                    name = subitem.text
            ingredients[unii] = name
    except: pass
    return ingredients

def extract_title(data, code):
    try:
        paragraphs = data.xpath("//v:code[@code='"+code+"']""/..//v:title",
                                namespaces=namespaces)
        text = list()
        text.append(' '.join(''.join(paragraphs[0].xpath(".//text()")).split()))
    except: text = ""
    return text

def extract_text(data, code):
    try:
        paragraphs = data.xpath("//v:code[@code='"+code+"']""/..//v:text"
                                " | //v:code[@code='"+code+"']""/..//v:title"
                                " | //v:code[@code='"+code+"']""/..//v:table"
                                " | //v:code[@code='"+code+"']""/..//v:paragraph"
                                " | //v:code[@code='"+code+"']/..//v:item",
                                namespaces=namespaces)
        text = list()
        for paragraph in paragraphs:
            text.append(' '.join(''.join(paragraph.xpath(".//text()")).split()))
    except: text = ""
    return text

def xml_parse(xfile):
    with gzip.open(xfile, 'rb') as xml_file:
        xml = xml_file.read()
        p = etree.XMLParser(huge_tree=True)
        data = etree.fromstring(xml, p)

        # sets
        if study_sets:
            setid = parse_set(data)
            print(xfile, setid, sep='\t', file=set_file)

        # approvals
        if study_approvals:
            approvals, codes = parse_approvals(data)
            for approval in approvals:
                print(xfile, approval, sep='\t', file=app_file)

        if study_ingredients:
            # active ingredients
            activeIngredients = parse_ingredients(data, 'active')
            for unii, name in activeIngredients.items():
                print(xfile, unii, name.lower(), sep='\t', file=ing_file)
    
            # inactive ingredients
            inactiveIngredients = parse_ingredients(data, 'inactive')
            for unii, name in inactiveIngredients.items():
                name = name.encode('utf8', errors="xmlcharrefreplace").decode('latin-1')
                print(xfile, unii, name.lower(), sep='\t', file=ina_file)

        # all text sections
        for (code, outfile) in zip(loincs, outfiles):
            if code == "34390-5" or code == "34391-3" or code == "50577-6" or code == "50578-4":
                text = extract_title(data, code)
            else:
                text = extract_text(data, code)
            text = '<br>'.join(OrderedSet(text))
            text = text.encode('utf8', errors="xmlcharrefreplace").decode('latin-1')
            if text:
                print(xfile, text,  sep='\t', file=outfile)

if __name__ == '__main__':
    loinc_file = "loinc.codes" # a file listing LOINC codes and what to call the output file for each
    dir = 'xmls'                       # where we expect the input xmls to be
    outdir = 'extracted'               # where the output will go
    ingredients_file = 'active_ingredients'
    inactive_file = 'inactive_ingredients'
    approvals_file = 'approvals'
    sets_file = 'sets'

    try: mkdir(outdir)
    except: pass
    
    # prepare output file for sets
    study_sets = False
    try:
        set_file = open(outdir+'/'+sets_file+'.txt', 'w')
        study_sets = True
    except: pass
    
    # prepare output file for approvals
    study_approvals = False
    try:
        app_file = open(outdir+'/'+approvals_file+'.txt', 'w')
        study_approvals = True
    except: pass
    
    # prepare output files for ingredient sections
    study_ingredients = False
    try:
        ing_file = open(outdir+'/'+ingredients_file+'.txt', 'w')
        ina_file = open(outdir+'/'+inactive_file+'.txt', 'w')
        study_ingredients = True
    except: pass
    
    # prepare output files for all the text sections
    loincs = list()
    outfiles = list()
    with open(loinc_file) as lf:
        for line in lf:
            if line == "###\n": break
            if line[0] == "#": continue
            null, null, code, null, null, title = line.rstrip().split("\t")
            try:
                file = open(outdir+'/'+title+'.txt', 'w')
                loincs.append(code)
                outfiles.append(file)
            except:
                pass
    
    if study_sets or study_approvals or study_ingredients or loincs:
        # extract content from all xmls
        bins = next(walk(dir), (None, None, []))[1]
        bins.sort()
        for bin in bins:
            #print(bin)
            files = next(walk(dir+'/'+bin), (None, None, []))[2]
            files.sort()
            for file in files:
                #print(dir+'/'+bin+'/'+file)
                xml_parse(dir+'/'+bin+'/'+file)

