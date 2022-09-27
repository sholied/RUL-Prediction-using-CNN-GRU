RUL (Remind Usefull Life) Prediction
==
RUl Prediction for turbo engine machine using CNN stacked with GRU.
--
Data that we use is CMAPSS Dataset, we can access from https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq
We build CNNGRU model with various number of layers CNN and GRU, for run this scripts:

command to run:
*python RUL_Prediction --model=none --input=none --epochs=none*
- model : there arw three model (single gru, stacked gru and cnn+gru) for default is single gru,
to choose stacked gru just  write "stacked_gru" and for cnn+gru is "cnn_gru)
- input : this the directory of CMAPSSDataset by default is ~/CMAPSSData/
- epochs : number of epohs for training model, integer type, default 50

for example:
python RUL_Prediction --model=cnn_gru --input=CMAPSSData --epochs=100
