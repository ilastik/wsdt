# wsdt

[![test](https://github.com/ilastik/wsdt/actions/workflows/test.yml/badge.svg)](https://github.com/ilastik/wsdt/actions/workflows/test.yml)

Repo for the watershed on distance transform.


## Installation for development:

We recommend development using `conda`/`mamba`:

```bash
> mamba create -n wsdt-dev -c conda-forge -c ilastik-forge python=3.7 numpy-allocation-tracking numpy networkx nose jupyter notebook pip
> mamba activate wsdt-dev
> pip install -e .
```

## Usage

For usage, see the two notebooks in the `examples` folder.

## Install with conda (into an existing environment):

```
$ conda install -c conda-forge -c ilastik-forge wsdt
```
