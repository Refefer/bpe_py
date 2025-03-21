BPE Py
===
This library implements train and tranformation steps for learning Byte-Pair Encoding on a corpus.  It's fast to train and fast to encode/decode as it uses Rust behind the scenes.

Installation
===
1. Ensure you have rust toolchain downloaded and installed.
2. pip install . 

Usage
===

```python
from bpe_py import BPE

bpe = BPE.learn_from_file("/path/to/corpus", max_vocab=10_000)
# bpe = BPE.learn_from_corpus(["list", "of sentences", "to learn against"], max_vocab=10_000)

sentence = "hello world"

# Encodes to a list of u32
sequence = bpe.encode_str(sentence) # sequence: List[Int]

# Decodes a sequence of codepoints to the original sentence
reconstructed_sentence bpe.decode_to_str(sequence)

assert sentence == reconstructed_sentence

# Save vocab to disk
with open('/path/to/vocab', 'w') as out:
    json.dump(bpe.vocab(), out)

# Save vocab to disk
with open('/path/to/vocab', 'w') as out:
    json.dump(bpe.vocab(), out)

# Load BPE from vocab
with open('/path/to/vocab') as out:
    vocab = json.load(out)
    bpe = BPE.load_from_vocab(vocab)
'''
