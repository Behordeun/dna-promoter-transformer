# DNA promoter transformer

A convolution-plus-transformer classifier for DNA promoter recognition. It runs on two real datasets: the classic E. coli promoter set and a genome-scale human non-TATA promoter set. The convolution stem detects short local motifs and hands those features to a transformer encoder, which classifies from a CLS token. Promoter signals are short boxes (-10, -35), so the conv gives the model the right local prior and makes even the ~100-example E. coli set learnable.

## Results

E. coli promoters (UCI, 106 sequences), 10-fold stratified cross-validation, every sequence held out once:

| Method | Errors / 106 |
| --- | --- |
| KBANN (Towell et al. 1990) | 4 |
| this model (conv + transformer) | 7 |
| standard backprop (1990) | 8 |
| nearest neighbour (k=3) | 13 |
| ID3 | 19 |

The 7/106 figure is deterministic at the default seed (`make ecoli` reproduces it).

Human non-TATA promoters (genomic-benchmarks, 36,131 sequences), trained on the Apple MPS GPU, evaluated on the authors' held-out test split of 9,034 sequences:

| Metric | Value |
| --- | --- |
| Accuracy | 0.885 |
| Precision | 0.927 |
| Recall | 0.855 |
| F1 | 0.890 |

These are the numbers in `checkpoints/metrics.json` from the run that produced the checkpoint. They are from a single training run with early stopping on a validation slice, so they move a little by seed. They sit in the range of the CNN baseline the benchmark authors report for this dataset.

## Datasets

- E. coli promoters: Harley, Reynolds & Noordewier (1987), UCI Machine Learning Repository, DOI 10.24432/C5S01D, CC BY 4.0. Downloaded by `make data`.
- Human non-TATA promoters: Grešová et al. (2023), "Genomic benchmarks: a collection of datasets for genomic sequence classification", BMC Genomic Data 24:25. Fetched by the `genomic-benchmarks` package, no account needed.

## Run it

```
make setup     # venv + install
make data      # download the E. coli set
make ecoli     # 10-fold cross-validation on E. coli
make human     # train on the human set (uses the GPU if present)
make test      # test suite
make lint      # ruff + mypy
```

`make human` picks the device automatically: Apple MPS first, then CUDA, then CPU. On an M3 Pro the human run finishes in a few minutes.

### Predict on a new sequence

Once you have a checkpoint from `make human`:

```
.venv/bin/python -m src.predict --checkpoint checkpoints/human.pt "ACGT...251bp..."
```

It prints the predicted class and the promoter probability.

## Layout

```
src/data.py          E. coli loader
src/data_human.py    human dataset loader
src/model.py         conv + transformer classifier
src/train.py         E. coli cross-validation
src/train_human.py   human training on GPU, saves a checkpoint
src/predict.py       classify a new sequence from a checkpoint
src/device.py        MPS / CUDA / CPU selection
tests/               data, model and evaluation tests
```
