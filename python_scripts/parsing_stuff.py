import spacy
from nltk.tokenize import sent_tokenize, word_tokenize


text = """She saw a secret little clearing, and a secret little hot made of rustic poles. And she had never been here before! She realized it was the quiet place where the growing pheasants were reared; the keeper in his shirtâ€‘sleeves was kneeling, hammering. The dog trotted forward with a short, sharp bark, and the keeper lifted his face suddenly and saw her. He had a startled look in his eyes.
He straightened himself and saluted, watching her in silence, as she came forward with weakening limbs. He resented the intrusion; he cherished his solitude as his only and last freedom in life.
"I wondered what the hammering was," she said, feeling weak and breathless, and a little afraid of him, as he looked so straight at her. "Ah'm getting the coops ready for the young bods,'" he said, in broad vernacular.
She did not know what to say, and she felt weak. "I should like to sit down a bit," she said. "Come and sit here," he said, going in front of her to the hut, pushing aside some timber and stuff, and drawing out a rustic chair, made of hazel sticks."""

def process_text(text):
	sentences = sent_tokenize(text)
	new_sentences = []
	print(sentences)
	for sentence in sentences: 
		tokens = word_tokenize(sentence)
		new_sentence = " ".join(tokens)
		new_sentences.append(new_sentence)
	return new_sentences

def get_word_order(sentences):
	has_svo = 0
	has_sov = 0
	has_vos = 0
	has_vso = 0
	has_ovs = 0
	has_osv = 0
	for sentence in sentences:
		nlp = spacy.load('en_core_web_lg')
		doc = nlp(sentence)
		print(doc)
		deps = []
		for token in doc:
			print(token.text, token.dep_)
			deps.append((str(token.text),str(token.dep_)))
		deps_all = " ".join([dep if text != "," and text != ";" else "separator" for text, dep in deps ])
		deps_all = deps_all.split("separator")
		deps_all = [item.split() for item in deps_all]
		print(deps_all)
		for item in deps_all:
			subj = None
			verb = None
			obj = None
			for dependency in item:
				if dependency in ["nsubj", "nsubjpass"]:
					print("subj found")
					subj = item.index(dependency)
				elif dependency == "ROOT":
					print("verb found")
					verb = item.index("ROOT")
				elif dependency == "dobj":
					print("obj found")
					obj = item.index("dobj")

			print(subj, verb, obj)

			if type(subj) == int and type(verb) == int and type(obj) == int:
				if subj < verb and subj < obj:
					if verb < obj:
						print("word order is SVO")
						has_svo += 1
					else:
						print("word order is SOV")
						has_sov += 1
				elif obj < verb and obj < subj:
					if verb < subj:
						print("word order is OVS")
						has_ovs += 1
					else:
						print("word order is OSV")
						has_osv += 1
				elif verb < obj and verb < subj:
					if obj < subj:
						print("word order is VOS")
						has_vos += 1
					else:
						print("word order is VSO")
						has_vso += 1
			else:
				print("sorry no order could be established")
	return [has_svo, has_sov, has_vos, has_vso, has_ovs, has_osv]

#get_word_order('He ate the apple.')
new_sentences = process_text(text)
print(get_word_order(new_sentences))

### take into account: multiple clauses (separated by commas, but commas can also separate list items etc.)
### take into account sentences with no direct object
## get sentences from text, then parse, then check punct items if , then split 

## sentence, split sentence on PUNCT, process each of these elements, then sum all of the found word orders (e.g. 1 SVO 1 SOV etc.)

## conj as verb?, ccomp as verb?
## also do SV as SVO? and SO as SOV?