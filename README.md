# EMIT GHG - Greenhouse Gas Detection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Point-source methane and CO2 detection from EMIT hyperspectral imaging data.

## Overview

This codebase implements a classical matched filter applied independently along each pushbroom column (Thompson et al., 2015, 2016; Frankenberg et al., 2016). Signatures are calculated on a scene-specific basis to account for local water vapor, elevation and solar position as in Foote et al. (2020). Statistical control for surface reflectance is used as in Elder et al. (2020).

**Note**: This is research code made available during active development for open science.

## Quick Start

### Installation

**Important**: GDAL must be installed via conda/mamba/pixi (it cannot be installed via pip alone).

#### Option 1: Using Pixi (Recommended)

[Pixi](https://pixi.sh) is a modern package manager that handles both conda and PyPI dependencies.

**Then install EMIT GHG:**
```bash
# Install
pixi install

# Verify installation
pixi run verify

# Enter the environment
pixi shell
```

#### Option 2: Using Mamba

```bash
# Create environment and install dependencies
mamba env create -f environment.yml
conda activate emit-ghg

# Install package in editable mode
pip install -e .
```


### Basic Usage

Process a single EMIT scene:

```bash
emit-ghg-process \
    radiance.img \
    obs.img \
    loc.img \
    glt.img \
    l1b_bandmask.img \
    l2a_mask.img \
    /output/path/scene_id \
    --state_subs state.img \
    --loglevel INFO
```

or directly run diffmf:

```
diffmf --help
```

### Running Tests

```bash
pytest
pytest --cov=emit_ghg
```

## References

- Thompson, D. R., et al. (2015). Real-time remote detection and measurement for airborne imaging spectroscopy: a case study with methane. *Atmospheric Measurement Techniques*, 8(10), 4383-4397.
- Frankenberg, C., et al. (2016). Airborne methane remote measurements reveal heavy-tail flux distribution in Four Corners region. *PNAS*, 113(35), 9734-9739.
- Thompson, D.R., et al. (2016). Space‐based remote imaging spectroscopy of the Aliso Canyon CH4 superemitter. *Geophysical Research Letters*, 43(12), 6571-6578.
- Foote, M.D., et al. (2020). Fast and accurate retrieval of methane concentration from imaging spectrometer data using sparsity prior. *IEEE TGRS*, 58(9), 6480-6492.
- Elder, C. D., et al. (2020). Airborne mapping reveals emergent power law of arctic methane emissions. *Geophysical Research Letters*, 47(3), e2019GL085707.
- Thorpe, A. K. et al. (2024). Attribution of individual methane and carbon dioxide emission sources using EMIT observations from space. *Science Advances*, 46(9), eadh2391.


