# Long Short-Term Memory (LSTM) for Modeling COVID-19 Cases in Tokyo, Japan

This repository contains a Python implementation of an LSTM, a type of Recurrent Neural Network first introduced by Hochreiter & Schmidhuber (1997) [1] and further modified by Gers *et al.* (2000) [2], to model COVID-19 cases in Tokyo, Japan. See `LSTM_theory.pdf` for implementation details.

## Files
- `src/lstm.py`: Python code
- `LSTM_theory.pdf`: Details of LSTM model
- `plots/*`: LSTM predictions & training loss visualized
- `data/*`: Training & validation data sets

## Dependencies

Code was tested on:
- Python 3.7.6
- NumPy 1.19.1
- Pandas 1.1.5

This implementation does not use Tensorflow or Keras.

## References
1. Hochreiter, S. & Schmidhuber, J. Long short-term memory. *Neural Computation* **9** 8, pp.1735–1780 (1997). https://doi.org/10.1162/neco.1997.9.8.1735
2. Gers, F.A., Schmidhuber, J.A. & Cummins, F.A. Learning to forget: Continual prediction with LSTM. *Neural Computation* **12** 10 (2000). https://doi.org/10.1162/089976600300015015