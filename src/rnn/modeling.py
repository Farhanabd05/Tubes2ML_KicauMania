def caption_steps(sequence_length: int) -> int:
    return int(sequence_length) - 1


def build_caption_model(is_lstm: bool, layers: int, hidden_state: int, vocab_size: int, sequence_length: int = 35):
    from tensorflow.keras.layers import Concatenate, Dense, Embedding, Input, LSTM, RepeatVector, SimpleRNN
    from tensorflow.keras.models import Model

    steps = caption_steps(sequence_length)
    if steps < 1:
        raise ValueError("sequence_length harus lebih besar dari 1.")

    image_input = Input(shape=(2048,), name="Image_Input")
    caption_input = Input(shape=(steps,), name="Caption_Input")

    image_context = Dense(256, activation="tanh", name="Image_Projection")(image_input)
    image_context = RepeatVector(steps, name="Image_Context_Repeat")(image_context)

    caption_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=256,
        mask_zero=False,
        name="Token_Embedding",
    )(caption_input)

    x = Concatenate(axis=-1, name="Context_Concat")([caption_embedding, image_context])
    recurrent_cls = LSTM if is_lstm else SimpleRNN
    prefix = "LSTM" if is_lstm else "RNN"

    for idx in range(layers):
        x = recurrent_cls(
            hidden_state,
            return_sequences=True,
            name=f"{prefix}_Layer_{idx + 1}",
        )(x)

    output = Dense(vocab_size, activation="softmax", name="Output_Layer")(x)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
    )
    return model


def greedy_decode_keras(keras_model, image_feature, text_util, max_len=35):
    import numpy as np

    stop_tokens = {"", "pad", "<pad>", "end", "<end>"}
    skip_tokens = {"unk", "<unk>", "start", "<start>"}
    start_token = text_util.word_to_idx.get("start", text_util.word_to_idx.get("<start>"))
    end_token = text_util.word_to_idx.get("end", text_util.word_to_idx.get("<end>"))
    if start_token is None or end_token is None:
        raise KeyError("Vocabulary harus punya token start dan end.")

    steps = caption_steps(text_util.sequence_length)
    input_seq = np.zeros((1, steps), dtype=np.int32)
    input_seq[0, 0] = start_token
    words = []

    for step in range(min(max_len, steps)):
        probs = keras_model.predict([image_feature.reshape(1, -1), input_seq], verbose=0)
        next_token = int(np.argmax(probs[0, step]))
        next_word = text_util.idx_to_word.get(next_token, "")

        if next_token == end_token or next_word in stop_tokens:
            break

        if next_word not in skip_tokens:
            words.append(next_word)
        if step + 1 < steps:
            input_seq[0, step + 1] = next_token

    return " ".join(words)
