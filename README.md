# SynthCtrl

A Python library for implementing Synthetic Control methods for causal inference, including the Classical Synthetic Control method and Synthetic Difference-in-Differences (SDID).

[![PyPI version](https://badge.fury.io/py/SynthCtrl.svg)](https://badge.fury.io/py/SynthCtrl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Synthetic Control is a statistical method used for comparative case studies. It constructs a weighted combination of control units to create a synthetic version of the treated unit, allowing for the estimation of causal effects in settings where a single unit receives treatment and multiple units remain untreated.

This library provides implementations of:

- **Classical Synthetic Control** (Abadie & Gardeazabal, 2003)
- **Synthetic Difference-in-Differences** (Arkhangelsky et al., 2021)

## Installation

```bash
pip install SynthCtrl
```

## Features

- Easy-to-use API with scikit-learn-like interfaces
- Automatic treatment date determination
- Bootstrap for statistical inference
- Comprehensive visualization tools
- Detailed documentation and examples

## Quick Start

```python
import pandas as pd
from synthetic_control import ClassicSyntheticControl

# Load data
data = pd.read_csv("california_smoking.csv")

# Initialize model
sc = ClassicSyntheticControl(
    data=data,
    metric="cigarettes",
    period_index="year",
    unit_id="state",
    treated="california",
    after_treatment="after_treatment"
)

# Fit model
sc.fit()

# Predict counterfactual values
predictions = sc.predict()

# Estimate treatment effect
effect = sc.estimate_effect()
print(f"Average Treatment Effect: {effect['att']:.4f}")

# Bootstrap for statistical inference
bootstrap_results = sc.bootstrap_effect()
print(f"Standard Error: {bootstrap_results['se']:.2f}")
print(f"95% CI: [{bootstrap_results['ci_lower']:.2f}, {bootstrap_results['ci_upper']:.2f}]")

# Visualize results
sc.plot_model_results(figsize=(14, 7), show=True)
```

### Using Synthetic Difference-in-Differences

```python
from synthetic_control import SyntheticDIDModel

# Initialize SDID model
sdid_model = SyntheticDIDModel(
    data=data,
    metric="cigarettes",
    period_index="year", 
    unit_id="state",
    treated="california",
    after_treatment="after_treatment"
)

# Fit model
sdid_model.fit()

# Visualize results
sdid_model.plot_model_results(figsize=(14, 7), show=True)
```

## Documentation

For detailed documentation, visit the [GitHub pages](https://github.com/123yaroslav/SynthCtrl).

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating various use cases:

- Basic usage with California smoking data
- Advanced features and customization
- Comparison of different methods

## Citation

If you use this library in your research, please cite:

```
@software{SynthCtrl_python,
  author = {Yaroslav Rogoza},
  title = {SynthCtrl: A Python Library for Causal Inference},
  year = {2023},
  url = {https://github.com/123yaroslav/SynthCtrl},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 