## Notes on paper

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)

### Core Problem

Mapping sequences to sequences. Canonical example: language translation (english -> hindi, english -> latex). Other examples: speech recognition (audio signal -> text), question-answering (text -> text)

### Approach

Use two RNNs (more precisely, the paper uses LSTMs)
1. using RNN1 (generally known as the encoder), ingest an input sequence and map to fixed-size vector (embedding)
2. using RNN2 (generally known as the decoder), initialize hidden state to fixed-size vector from step 1 and generate a sequence

### Benchmarks

English to French translation from [WMT 14 dataset](https://torchtext.readthedocs.io/en/latest/datasets.html#wmt14)

Result: BLEU score = 34.8 with penalization on out-of-vocabulary words

### Studies

1. Effect of sequence length
2. Effect of reversing input sequence
3. Effect of depth in RNNs
4. Embedding (fixed-size vector) structure for similar sentences written differently (active vs passive voice)

### Details and comments

* "DNNs are powerful because they can perform arbitrary parallel computation for a modest number of steps"
* "A surprising example of the power of DNNs is their ability to sort N N-bit numbers using only 2 hidden layers of quadratic size" [paper](https://people.cs.uchicago.edu/~razborov/files/helsinki.pdf)

* [Architecture] Pair of RNNs:
  * Encoder: Use RNN to step through input sequence and final hidden state is sequence representation/embedding
  * Decoder: Second RNN is essentially a language model (generates one token at a time) conditioned on i.e. uses as input, the representation from the encoder.

* Basic idea: ![Seq2Seq Anatomy](https://github.com/TreeinRandomForest/RHCourse/blob/main/p3-seq2seq/media/seq2seq-arch.png)

* [Starting output generation] <EOS> is a special token that marks the end of the input sequence and the start of the output sequence.

* [Language Translation Task] BLEU score on WMT'14 English->French: 34.81 from ensemble of 5 deep LSTMs (384M parameters, 8000 dimensional embedding) with input sequence read left-to-right + beam search (more below)

* BLEU score from SMT (Statistical Machine Translation - not based on NNs) - 33.30

* Vocabulary size: 80k words + penalization when target output sequence contained word not in vocabulary

* Improvement in BLEU score by doing top-1000 prediction (BLEU = 36.5)

* [Reverse input sequences] No/minimal degradation when input sequences are long. Input/source sequences were reversed to reduce distance between word position in input sequence and word position in output sequence.

* [Embedding structure] Similar input sequences map to close-by embeddings. Embeddings approximately invariance to active vs passive voice (since this doesn't change the meaning which is relevant for translation).

* [Why not use just one RNN] Works when there's alignment between input and output sequence i.e. they have same lengths and the kth output token can be predicted after reading the first-k input tokens.

* [Why LSTMs] RNNs have problems learning long-range dependencies. LSTMs are designed explicitly for this purpose (but we'll use both in our implementation)

### Special Topics

#### BLEU Score

#### Beam Search and various decoding strategies (top-k and top-p)

### Exercises

1. (Shrey) Look at distributions of gradients close to L1 and far from L1