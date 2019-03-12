# csc2516-project
CSC2516 project on data-driven confusion classification learning


# TODOs:
1. Maybe augmented levenstein distance (at least the logic part).
- Easier to compare with the literature.
- Easier to argue that we are building on existing knowledge.

2. NN: Architecture should be a bidirectional LSTM, perhaps with an attention mechanism.
- RNN that predicts phoneme error rate (or word error rate).
- RNN that predicts, for each phoneme, whether it will be misheard.
- RNN that predicts, for each phoneme, a softmax over all phonemes predicting what the output phoneme will be.

3. Find baselines (preferably in the literature). Our own baselines:
- MLE for each of the three models. See picture on FB.

4. Post processing of the results to get nice tables of results and such.
- Maybe some visualization as well.
