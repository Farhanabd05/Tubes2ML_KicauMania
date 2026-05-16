# RNN/LSTM Experiment Summary

- Architecture: `preinject_v2`.
- Model follows the assignment pre-inject design: `Image_Projection` is inserted as timestep `t=-1` before `<start>`, then `Drop_Image_Timestep` aligns recurrent outputs with shifted caption targets.
- Best scratch model: SimpleRNN Shallow_Large with BLEU-4=0.1325.
- Best model diversity: 88 unique captions; top-caption frequency=6.00%.
- Deeper/larger models should be judged with BLEU/METEOR and inference time, not loss alone.
- Keras vs scratch comparison is saved in `keras_vs_scratch.csv`; small score differences usually come from decoding and numerical details.
- Best max caption length in sweep: 10 with BLEU-4=0.1392.
- Preinject-v2 smoke checks cover the 34-step caption input/output shape, t=-1 image timestep, and sample-weight pad masking.
- LSTM is expected to handle longer dependencies better than SimpleRNN because gates reduce vanishing-gradient effects.
- Use `qualitative_samples.csv` for the 10-image high/medium/low qualitative comparison in the report.
