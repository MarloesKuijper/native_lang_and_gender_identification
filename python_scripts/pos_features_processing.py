import argparse
import conllu
from conllu import parse, parse_tree
import os, re

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp","--fp", type=str, help="Filepath")
    args = parser.parse_args()
    return args


def get_tree(files, out):
	for file in files:
		with open(args.fp+file, "r", encoding="utf-8") as infile, open(out, "a", encoding="utf-8") as outfile:
			data = infile.read()
			items = [item for item in data.split("\n\n")]
			if len(items) > 5000:
				threshold = 5000
			else:
				threshold = len(items)
			for item in items[:threshold]:
				try:
					parsed = parse(item)
					item_postags = []
					if parsed:
						for item in parsed[0]:
							item_postags.append(item["upostag"])
						outfile.write(" ".join(item_postags)+", " + file.split("-")[0]+"\n")
				except conllu.parser.ParseException:
					print(file.split("-")[0])
					print(item)


	# get pos tag data per country
	# do some post-processing to make sure all countries use the same tags (and preferably the same tags as NLTK)
	# incorporate into an SVM-like model

def get_sentences(files, out):
	for file in files:
		with open(args.fp+file, "r", encoding="utf-8") as infile, open(out, "a", encoding="utf-8") as outfile:
			data = infile.read()
			items = [item for item in data.split("\n\n")]
			for item in items:
				#print(item)
				for line in item.split("\n"):
					if line.startswith("#SENT"):
						#print(line)
						line = line[6:]
						line = re.sub('<c>[^<]+?</c>', '', line)
						line = re.sub('<[^<]+?>', '', line)
						line = " ".join(line.split())
						outfile.write(line)
						outfile.write("\n")
						print(line)


if __name__ == "__main__":
    args = create_arg_parser()
    files = [item for item in os.listdir(args.fp) if item.endswith("conllu")]
    print(files)
    get_sentences(files, args.fp+"sentences.txt")
    #get_tree(files, args.fp+"pos_tag_distributions.txt")
