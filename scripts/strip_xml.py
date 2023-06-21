import argparse
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="path to input file")
parser.add_argument("--output", type=str, help="path to output file")
args = parser.parse_args()

with open(args.input,'r') as fi:
    content = fi.read()

soup = BeautifulSoup(content, features="xml")
with open(args.output,'w') as fo:
    for item in soup.find_all('seg'):
        lines_in_item = item.text.split('\n')
        [fo.write(x.strip() + '\n') for x in lines_in_item if x.strip() != ""]
