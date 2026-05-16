# RNN/LSTM Experiment Summary

- Architecture: `context_v2`.
- Best scratch model: SimpleRNN Shallow_Large with BLEU-4=0.1395.
- Best model diversity: 95 unique captions; top-caption frequency=2.00%.
- Deeper/larger models should be judged with BLEU/METEOR and inference time, not loss alone.
- Keras vs scratch comparison is saved in `keras_vs_scratch.csv`; small score differences usually come from decoding and numerical details.
- Best max caption length in sweep: 10 with BLEU-4=0.1484.
- Context-v2 smoke checks cover the 34-step caption input/output shape and sample-weight pad masking.
- LSTM is expected to handle longer dependencies better than SimpleRNN because gates reduce vanishing-gradient effects.
- Use `qualitative_samples.csv` for the 10-image high/medium/low qualitative comparison in the report.
