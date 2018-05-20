import os
import sys
import requests
import json
import csv
from glob import glob

parser_url = 'http://localhost:8080/relationExtraction/text'
parsing_header = {'Accept': 'application/json'}

def parse(text):
  params = {'doCoreference': True, 'isolateSentences': False, 'text': text}
  r = requests.post(parser_url, json=params)
  return r.json()

if __name__ == '__main__':
  with open('/media/jun/USB/all-the-news/articles1.csv', 'rb') as f:
    reader = csv.reader(f)
    header = reader.next()
    for r in reader:
      id = r[1]
      outf = '/media/jun/USB/all-the-news/parsed/{}.json'.format(id)
      if os.path.exists(outf):
        print 'Skip', id
        continue
      
      print 'Processing', id
      parsed = parse(r[-1])
      with open(outf, 'w') as o:
        json.dump(parsed, o)

  '''
  for root, subdirs, _ in os.walk(sys.argv[1]):
    for subdir in subdirs:
      folder = os.path.join(root, subdir)
      for fn in glob(os.path.join(folder, '*.txt')):
        outf = os.path.join(folder, os.path.basename(fn) + '.json')
        if os.path.exists(outf):
          print 'Skip', outf
          continue
        with open(fn, 'r') as f, open(outf, 'w') as o:
          print 'Processing', fn
          parsed = parse(f.read())
          json.dump(parsed, o)
  '''         