# Fourier Neural Operator

The FNO comparison uses [NeuralOperator](https://github.com/neuraloperator/neuraloperator) at commit `00b7d86f8d74ff0af55da53eb585fe26df9c71f0`.

From the NCRF repository root, prepare the source at the expected location:

```bash
python third_party/fourier_neural_operator/setup_fno.py
```

This command requires Git and network access. It verifies the commit and the SHA-256 hash of `neuralop/models/fno.py` after line-ending normalization before copying the `neuralop` package into this directory. It will not overwrite an existing `neuralop` directory. Install the FNO dependencies specified in the main project's `pyproject.toml`.

NeuralOperator is distributed under the MIT license; see `LICENSE`.
