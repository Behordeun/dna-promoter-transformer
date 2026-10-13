# DNA promoter transformer

A convolution-plus-transformer classifier for DNA promoter recognition, trained
and evaluated on two real datasets: the classic E. coli promoter set and a
genome-scale human non-TATA promoter set. The convolution stem acts as a motif
detector and gives the model the local inductive bias that short promoter
signals need; the transformer encoder reads those features and classifies from
a CLS token.

The point of the project is a clean, honest sequence-classification pipeline:
real labelled data, a model that suits the data size, and evaluation that does
not cut corners.

## Results

E. coli promoters (UCI, 106 sequences), 10-fold stratified cross-validation,
every sequence held out once:

| Method | Errors / 106 |
| --- | --- |
| KBANN (Towell et al. 1990) | 4 |
| this model (conv + transformer) | 9 |
| standard backprop (1990) | 8 |
| nearest neighbour (k=3) | 13 |
| ID3 | 19 |

Human non-TATA promoters (genomic-benchmarks, 36,131 sequences), trained on the
Apple MPS GPU, evaluated on the authors' held-out test split of 9,034 sequences:

| Metric | Value |
| --- | --- |
| Accuracy | 0.889 |
| Precision | 0.916 |
| Recall | 0.875 |
| F1 | 0.895 |

The human numbers come from a single training run with early stopping on a
validation slice; they will vary a little by seed. They sit in the same range
as the CNN baseline the benchmark authors report for this dataset.

## Datasets

- E. coli promoters: Harley, Reynolds & Noordewier (1987), UCI Machine Learning
  Repository, DOI 10.24432/C5S01D, CC BY 4.0. Downloaded by `make data`.
- Human non-TATA promoters: Grešová et al. (2023), "Genomic benchmarks: a
  collection of datasets for genomic sequence classification", BMC Genomic Data
  24:25. Fetched by the `genomic-benchmarks` package, no account needed.

## Run it

```
make setup     # venv + install
make data      # download the E. coli set
make ecoli     # 10-fold cross-validation on E. coli
make human     # train on the human set (uses the GPU if present)
make test      # test suite
```

`make human` selects the device automatically: Apple MPS first, then CUDA, then
CPU. On an M3 Pro the human run finishes in a few minutes.

## Layout

```
src/data.py          E. coli loader
src/data_human.py    human dataset loader
src/model.py         conv + transformer classifier
src/train.py         E. coli cross-validation
src/train_human.py   human training on GPU
src/device.py        MPS / CUDA / CPU selection
tests/               data, model and evaluation tests
```
