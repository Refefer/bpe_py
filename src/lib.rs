mod progress;

use std::borrow::Borrow;
use std::fmt::Write;
use std::fs::File;
use std::io::{Result as IOResult,BufReader,BufRead};
use std::collections::HashMap;

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use rayon::prelude::*;

use pyo3::prelude::*;
use pyo3::exceptions::{PyUnicodeDecodeError};

use crate::progress::CLProgressBar;

pub fn open_file_for_reading(path: &str) -> IOResult<Box<dyn BufRead>> {
    let f = File::open(path)?;

    let f = BufReader::new(f);
    let result: Box<dyn BufRead> = if path.ends_with(".gz") {
        let decoder = BufReader::new(GzDecoder::new(f));
        Box::new(decoder)
    } else {
        Box::new(f)
    };
    Ok(result)
}

type Vocab<A> = Vec<Vec<A>>;
struct BPEEncoder {
    vocab: Vocab<u32>,
    lookup: HashMap<Vec<u8>, u32>,
    max_key_len: usize
}

impl BPEEncoder {
    pub fn decode_codepoint(vocab: &Vocab<u32>, codepoint: u32, into: &mut Vec<u8>) {
        if codepoint < 256 {
            into.push(codepoint as u8);
        } else {
            let pair = &vocab[codepoint as usize];
            BPEEncoder::decode_codepoint(vocab, pair[0], into);
            BPEEncoder::decode_codepoint(vocab, pair[1], into);
        }
    }

    pub fn new(vocab: Vocab<u32>) -> BPEEncoder {
        let mut lookup = HashMap::new();
        let mut max_key_len = 0;
        for i in 0..vocab.len() {
            let mut key = Vec::new();
            BPEEncoder::decode_codepoint(&vocab, i as u32, &mut key);
            max_key_len = max_key_len.max(key.len());
            lookup.insert(key, i as u32);
        }
        BPEEncoder { vocab, lookup, max_key_len }
    }

    pub fn encode(&self, sequence: &[u8]) -> Vec<u32> {
        let mut output = Vec::new();
        let mut i = 0;
        let s_l = sequence.len();
        while i < s_l {
            let max_j = (s_l - i).min(self.max_key_len+1);
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
    pub fn decode(&self, sequence: &[u32]) -> Vec<u8> {
        let mut buff = Vec::new();
        for s in sequence {
            BPEEncoder::decode_codepoint(&self.vocab, *s, &mut buff)
        }
        buff
    }
}

#[pyclass]
pub struct BPE {
    encoder: BPEEncoder
}

impl BPE {

    pub fn count_pairs(
        chunks: &[Vec<u32>],
        chunk_size: Option<usize>
    ) -> HashMap<(u32, u32), usize> {
        let chunk_size = chunk_size.unwrap_or(1_000);
        let stats = (0..chunks.len()).into_par_iter().step_by(chunk_size).map(|start_idx| {
            let end_idx = (start_idx + chunk_size).min(chunks.len());
            let mut hm = HashMap::new();
            chunks[start_idx..end_idx].iter().for_each(|chunk| {
                chunk.windows(2).for_each(|pair| { 
                    let p = (pair[0], pair[1]);
                    *hm.entry(p).or_insert(0) += 1;
                });
            });
            hm
        }).reduce(||HashMap::new(), |mut hm1, hm2| {
            hm2.into_iter().for_each(|(k, v)| {
                let e = hm1.entry(k).or_insert(0); 
                *e = *e + v;
            });
            hm1
        });
        stats
    }

    pub fn merge(
        chunks: &mut [Vec<u32>],
        pair: (u32, u32),
        new_idx: u32
    ) {
        chunks.par_iter_mut().for_each(|mut chunk| {
            let mut buffer = Vec::with_capacity(0);
            let mut mutated = false;
            // Skip tells the iterator to skip the next token because it was collapsed
            let mut skip = false;
            chunk.windows(2).enumerate().for_each(|(i, p)| {
                if !skip {
                    if p[0] == pair.0 && p[1] == pair.1 {
                        if !mutated {
                            buffer.extend_from_slice(&chunk[0..i]);
                        }
                        mutated = true;
                        skip = true;
                        buffer.push(new_idx);
                    } else if mutated {
                        buffer.push(p[0]);
                    }
                } else {
                    skip = false;
                }
            });

            if mutated {
                std::mem::swap(&mut buffer, &mut chunk);
            }
        });
    }

    fn convert_text_to_chunks(
        text: Vec<String>
    ) -> Vec<Vec<u32>> {
        text.into_par_iter().map(|line| {
            line.as_bytes().iter().map(|c| *c as u32).collect()
        }).collect()
    }

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
        while dictionary.len() < max_vocab {
            let counts = BPE::count_pairs(&chunks, None);
            if let Some((pair, _cnt)) = counts.iter().max_by_key(|(_pair, count)| *count) {
                BPE::merge(&mut chunks, *pair, dictionary.len() as u32);
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
    
    #[staticmethod]
    pub fn learn_from_corpus(
        text: Vec<String>,
        max_vocab: usize
    ) -> BPE {
        let vocab = BPE::learn_vocab(text, max_vocab);
        let encoder = BPEEncoder::new(vocab);
        BPE { encoder }
    }

    #[staticmethod]
    pub fn load_from_vocab(
        vocab: Vocab<u32>,
    ) -> BPE {
        let encoder = BPEEncoder::new(vocab);
        BPE { encoder }
    }


    pub fn vocab(&self) -> Vec<Vec<u32>> {
        self.encoder.vocab.clone()
    }

    /*
    #[staticmethod]
    pub fn learn_from_file(
        path: String
    ) -> IOResult<BPE> {
        let s = open_file_for_reading(path)?;
        BPE { bpe: vocab }
    }
    */

    pub fn encode_str(&self, input: &str) -> Vec<u32> {
        self.encoder.encode(input.as_bytes())
    }

    pub fn encode_array(&self, input: Vec<u8>) -> Vec<u32> {
        self.encoder.encode(input.as_slice())
    }

    pub fn decode_to_bytes(&self, input: Vec<u32>) -> Vec<u8> {
        self.encoder.decode(input.as_slice())
    }

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
