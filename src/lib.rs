mod progress;

use std::fmt::Write;
use std::fs::File;
use std::io::{Result as IOResult,BufReader,BufRead};
use std::collections::{HashMap,HashSet};

use flate2::read::GzDecoder;
use rayon::prelude::*;
use dashmap::DashMap;

use pyo3::prelude::*;
use pyo3::exceptions::{PyUnicodeDecodeError};

use crate::progress::CLProgressBar;

/**
 * Reads a corpus from disk.  It handles gzip versus uncompressed natively.
 *
 */
fn read_corpus(path: &str) -> IOResult<Vec<String>> {
    let f = File::open(path)?;
    let f = BufReader::new(f);
    let mut corpus = Vec::new();
    if path.ends_with(".gz") {
        let decoder = BufReader::new(GzDecoder::new(f));
        for line in decoder.lines() {
            let line = line?;
            corpus.push(line);
        }
    } else {
        for line in f.lines() {
            let line = line?;
            corpus.push(line);
        }
    };
    Ok(corpus)
}

type Vocab<A> = Vec<Vec<A>>;

/**
 * The main magic.  It encodes and decodes strings from a learned BPE vocabulary.
 * */
struct BPEEncoder {
    /// Code points for underlying text.
    vocab: Vocab<u32>,

    /// Maps vectors of characters to codepoints
    lookup: HashMap<Vec<u8>, u32>,

    /// Encoding takes advantage of trying to maximally match a sequence by starting at the longest
    /// potential sequence.
    max_key_len: [usize; 256]
}

impl BPEEncoder {
    /** 
     * Decodes a codepoint into its original representation.
     */
    pub fn decode_codepoint(
        // Learned vocabulary
        vocab: &Vocab<u32>, 

        // Codepoint
        codepoint: u32, 

        // Where to store the decoded string.  It does _not_ clear the vector before writing,
        // allowing the user to stream tokens into it.
        into: &mut Vec<u8>
    ) {
        if codepoint < 256 {
            into.push(codepoint as u8);
        } else {
            let pair = &vocab[codepoint as usize];
            BPEEncoder::decode_codepoint(vocab, pair[0], into);
            BPEEncoder::decode_codepoint(vocab, pair[1], into);
        }
    }

    /**
     * Creates a new BPEEncoder from a given vocab.
     */
    pub fn new(
        vocab: Vocab<u32>
    ) -> BPEEncoder {
        let mut lookup = HashMap::new();
        let mut max_key_len = [1; 256];
        for i in 0..vocab.len() {
            let mut key = Vec::new();
            BPEEncoder::decode_codepoint(&vocab, i as u32, &mut key);
            let c = key[0] as usize;
            max_key_len[c] = max_key_len[c].max(key.len());
            lookup.insert(key, i as u32);
        }
        BPEEncoder { vocab, lookup, max_key_len }
    }

    /**
     * Given as sequence of characters, encode it into a set of 32bit codepoints.
     */
    pub fn encode(
        &self, 
        sequence: &[u8]
    ) -> Vec<u32> {
        let mut output = Vec::new();
        let mut i = 0;
        let s_len = sequence.len();
        while i < s_len {
            let max_key_len = self.max_key_len[sequence[i] as usize];
            let max_j = (s_len - i).min(max_key_len + 1);
            for j in (1..max_j+1).rev() {
                let end_point = i + j;
                let candidate = &sequence[i..end_point]; 
                if let Some(codepoint) = self.lookup.get::<[u8]>(candidate) {
                    output.push(*codepoint);
                    i += j;
                    break
                }
            }
        }
        output
    }

    /**
     * Decodes a list of codepoints into the original underlying sequence.
     */
    pub fn decode(
        &self, 
        sequence: &[u32]
    ) -> Vec<u8> {
        let mut buff = Vec::new();
        for s in sequence {
            BPEEncoder::decode_codepoint(&self.vocab, *s, &mut buff)
        }
        buff
    }

    fn len(&self) -> usize {
        self.vocab.len()
    }
}

/// To speed up replacement, we create an inverted index which maps
/// pairs of sequences to the corpus line.
type InvertedIndex = DashMap<(u32, u32), HashSet<usize>>;

/**
 * BPE provides three main capabilities:
 * 1. Learn a byte-pair encoding from a corpus.
 * 2. Encode a string of text into a byte-pair encoded representation.
 * 3. Decode a byte-pair encoded representation back into original sequence
 */
#[pyclass]
pub struct BPE {
    encoder: BPEEncoder
}

impl BPE {

    /**
     * Counts codepoints in a corpus by pairs, creating an inverted index.
     */
    fn count_pairs(

        // Corpus to count
        chunks: &[Vec<u32>],

        // Number of chunks to perform in parallel.  
        chunk_size: Option<usize>

    ) -> InvertedIndex {
        let chunk_size = chunk_size.unwrap_or(1_000);
        let stats = (0..chunks.len()).into_par_iter().step_by(chunk_size).map(|start_idx| {
            let end_idx = (start_idx + chunk_size).min(chunks.len());
            let hm = DashMap::new();
            for doc_id in start_idx..end_idx {
                let chunk = &chunks[doc_id];
                chunk.windows(2).for_each(|pair| { 
                    let p = (pair[0], pair[1]);
                    let e = hm.entry(p);
                    e.or_insert_with(||HashSet::new()).insert(doc_id);
                });
            }
            hm
        }).reduce(||DashMap::new(), |hm1, hm2| {
            hm2.into_iter().for_each(|(k, v)| {
                if !hm1.contains_key(&k) {
                    hm1.insert(k, v);
                } else {
                    let mut v1 = hm1.entry(k).or_insert_with(||HashSet::new()); 
                    v.into_iter().for_each(|v_i| {
                        v1.insert(v_i);
                    });
                }
            });
            hm1
        });
        stats
    }

    fn __len__(&self) -> usize {
        self.encoder.len()
    }

    /**
     *  Adds a document to the inverted index.
     */
    fn add_chunk_to_index(
        inverted_idx: &InvertedIndex, 
        doc_id: usize, 
        chunk: &[u32]
    ) {
        chunk.par_iter().zip(chunk.par_iter().skip(1)).for_each(|pair| {
            let p = (*pair.0, *pair.1);
            let mut e = inverted_idx.entry(p).or_insert_with(|| HashSet::new());
            e.insert(doc_id);
        });
    }

    /**
     *  Removes a document to the inverted index.
     */
    fn remove_chunk_from_index(
        inverted_idx: &InvertedIndex, 
        doc_id: usize, 
        chunk: &[u32]
    ) {
        chunk.par_iter().zip(chunk.par_iter().skip(1)).for_each(|pair| {
            if let Some(mut hs) = inverted_idx.get_mut(&(*pair.0, *pair.1)) {
                hs.remove(&doc_id);
            }
        });
    }

    /**
     * Merges two codepoints into a new codepoint and updates the inverted index.
     */
    pub fn merge(
        inverted_idx: &mut InvertedIndex,
        chunks: &mut [Vec<u32>],
        doc_ids: &HashSet<usize>,
        pair: (u32, u32),
        new_idx: u32
    ) {
        let merged: Vec<_> = doc_ids.par_iter().map(|doc_id| {
            let chunk: &[u32] = &chunks[*doc_id];
            let mut buffer = Vec::with_capacity(chunk.len());
            // Skip tells the iterator to skip the next token because it was collapsed
            let mut skip = false;
            chunk.windows(2).for_each(|p| {
                if !skip {
                    if p[0] == pair.0 && p[1] == pair.1 {
                        skip = true;
                        buffer.push(new_idx);
                    } else {
                        buffer.push(p[0]);
                    }
                } else {
                    skip = false;
                }
            });
            (*doc_id, buffer)
        }).collect();

        merged.par_iter().for_each(|(doc_id, new_buff)| {
            let chunk = &chunks[*doc_id];
            BPE::remove_chunk_from_index(inverted_idx, *doc_id, chunk);
            BPE::add_chunk_to_index(inverted_idx, *doc_id, new_buff.as_slice());
        });

        merged.into_iter().for_each(|(doc_id, mut new_buff)| {
            std::mem::swap(&mut new_buff, &mut chunks[doc_id]);
        });

    }

    /**
     * Converts sequences of strings into a sequence of u8s
     */
    fn convert_text_to_chunks(
        text: Vec<String>
    ) -> Vec<Vec<u32>> {
        text.into_par_iter().map(|line| {
            line.as_bytes().iter().map(|c| *c as u32).collect()
        }).collect()
    }

    /**
     * Main algorithm for learning BPE.  It iteratively takes pairs of codepoints and merges the
     * most frequence ones together.  It performs this up until max_vocab size.
     */
    fn learn_vocab(
        text: Vec<String>,
        max_vocab: usize
    ) -> Vec<Vec<u32>> {
        
        let pb = CLProgressBar::new(max_vocab as u64 - 256, true);
        pb.update_message(|msg| { write!(msg, "Learning Vocab...").expect("Should never hit"); });

        // Convert string lines into characters
        let mut chunks = BPE::convert_text_to_chunks(text);

        // Insert the original 256 characters into the dictionary
        let mut dictionary: Vec<Vec<u32>> = (0..256u32).map(|c| vec![c]).collect();
        let mut inverted_idx = BPE::count_pairs(&chunks, None);

        // Until we've learned the maximum vocab size, keep performing
        while dictionary.len() < max_vocab {
            // This linear scan of the best pairs is one of the bottlenecks we run into.
            // There is likely a faster way to do this but haven't thought it through enough :)
            let best_key = inverted_idx.par_iter()
                .max_by_key(|kv| kv.value().len())
                .map(|kv| {
                    let (k, v) = kv.pair();
                    (k.clone(), v.clone())
                });

            if let Some((pair, doc_ids)) = best_key {
                BPE::merge(&mut inverted_idx, &mut chunks, &doc_ids, pair, dictionary.len() as u32);
                dictionary.push(vec![pair.0, pair.1]);
                pb.inc(1);
            }
        }
        pb.finish();
        dictionary
    }

}

#[pymethods]
impl BPE {
    
    /**
     * Learns a new BPE encoder from a corpus.
     */
    #[staticmethod]
    pub fn learn_from_corpus(
        text: Vec<String>,
        max_vocab: usize
    ) -> BPE {
        let vocab = BPE::learn_vocab(text, max_vocab);
        let encoder = BPEEncoder::new(vocab);
        BPE { encoder }
    }

    /**
     * Constructs a new encoder based on a previously learned vocabulary
     */
    #[staticmethod]
    pub fn load_from_vocab(
        vocab: Vocab<u32>,
    ) -> BPE {
        let encoder = BPEEncoder::new(vocab);
        BPE { encoder }
    }

    /**
     * Returns the compression table
     */
    pub fn vocab(&self) -> Vec<Vec<u32>> {
        self.encoder.vocab.clone()
    }

    /**
     * Learns a new BPE encoder from a corpus stored on disk.
     */
    #[staticmethod]
    pub fn learn_from_file(
        path: &str,
        max_vocab: usize
    ) -> PyResult<BPE> {
        let text = read_corpus(&path)?;
        let vocab = BPE::learn_vocab(text, max_vocab);
        let encoder = BPEEncoder::new(vocab);
        Ok(BPE { encoder })
    }

    /**
     * Encodes a string into a BPE encoded vector
     */
    pub fn encode_str(&self, input: &str) -> Vec<u32> {
        self.encoder.encode(input.as_bytes())
    }

    /**
     * Encodes a list of string into a BPE encoded vector.
     */
    pub fn encode_strs(&self, input: Vec<String>) -> Vec<Vec<u32>> {
        input.into_par_iter().map(|s| {
            self.encoder.encode(s.as_bytes())
        }).collect()
    }

    /**
     * Encodes a byte array into a BPE encoded vector.
     */
    pub fn encode_byte_array(&self, input: Vec<u8>) -> Vec<u32> {
        self.encoder.encode(input.as_slice())
    }

    /**
     * Decodes a BPE vector into a bytearray
     */
    pub fn decode_to_bytes(&self, input: Vec<u32>) -> Vec<u8> {
        self.encoder.decode(input.as_slice())
    }

    /**
     * Decodes a BPE vector into a string
     */
    pub fn decode_to_str(&self, input: Vec<u32>) -> PyResult<String> {
        let sequence = self.encoder.decode(input.as_slice());
        match std::str::from_utf8(sequence.as_slice()) {
            Ok(s) => Ok(s.to_string()),
            Err(_err) => {
                let err_msg = format!("Error creating string from {:?}", sequence);
                Err(PyUnicodeDecodeError::new_err(err_msg))
            }
        }
    }


}

#[pymodule]
fn bpe_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BPE>()?;
    Ok(())
}
