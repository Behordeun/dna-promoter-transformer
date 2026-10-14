# Contributing

Thanks for taking a look.

## Setup

```
make setup
make data
make test
make lint
```

`make lint` runs ruff and mypy strict. `make test` runs the E. coli suite, which trains the real model a few times, so it takes about half a minute.

## Before a pull request

- `make lint` and `make test` both pass.
- New behaviour comes with a test.
- Keep the conv-plus-transformer design intact. If you change the model's input length or vocabulary, the forward pass checks that `pos_embed` still matches, so run the shape tests.

## Data

The E. coli set downloads from UCI (`make data`). The human set is fetched by the `genomic-benchmarks` package on first run of `make human`; it needs no account. Neither dataset is committed.
