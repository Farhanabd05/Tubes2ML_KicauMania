import numpy as np
from pathlib import Path
try:
    from .layers import DenseOutputLayer, DenseProjectionLayer, EmbeddingLayer, LSTMLayer, RNNLayer
except ImportError:
    from layers import DenseOutputLayer, DenseProjectionLayer, EmbeddingLayer, LSTMLayer, RNNLayer

class ImageCaptioningModel:
    def __init__(self, keras_model, text_util, is_lstm=True):
        self.text_util = text_util
        self.is_lstm = is_lstm
        
        self.embedding = EmbeddingLayer()
        self.dense_projection = DenseProjectionLayer(2048, 256)
        self.recurrent_layers = []
        self.dense_output = DenseOutputLayer(256, text_util.vocab_size)
        
        self._load_weights(keras_model)
    
    def _load_weights(self, keras_model):
        for layer in keras_model.layers:
            if 'embedding' in layer.name:
                emb_weights = layer.get_weights()[0]
                self.embedding.set_w(emb_weights)
                break

        for layer in keras_model.layers:
            weights = layer.get_weights()
            if len(weights) == 2 and weights[0].shape == (2048, 256):
                self.dense_projection.set_weights(weights[0], weights[1])
                break
        
        layer_idx = 1
        while True:
            if self.is_lstm:
                layer_name = f"LSTM_Layer_{layer_idx}"
                LayerClass = LSTMLayer
            else:
                layer_name = f"RNN_Layer_{layer_idx}"
                LayerClass = RNNLayer
            
            try:
                layer = keras_model.get_layer(layer_name)
                weights = layer.get_weights()
                
                recurrent_unit = LayerClass()
                
                recurrent_unit.set_w(
                    units=layer.units,
                    kernel=weights[0],
                    recurrent_kernel=weights[1],
                    bias=weights[2]
                )
                
                self.recurrent_layers.append(recurrent_unit)
                layer_idx += 1
            except ValueError:
                break
        
        out_weights = keras_model.get_layer("Output_Layer").get_weights()
        self.dense_output.set_weights(out_weights[0], out_weights[1])        
    
    def reset_states(self):
        for layer in self.recurrent_layers:
            layer.ht = np.zeros(layer.units)
            if self.is_lstm:
                layer.ct = np.zeros(layer.units)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def forward_step(self, input_t):
        x = input_t
        for recurrent_layer in self.recurrent_layers:
            x = recurrent_layer.forward(x)
        logits = self.dense_output.forward(x.reshape(1, -1))
        probs = self.softmax(logits[0])
        return probs
    
    def generate_caption(self, image_feature, max_len=35):
        self.reset_states()
        
        x_minus1 = self.dense_projection.forward(
            image_feature.reshape(1, -1)
        )[0]
        
        self.forward_step(x_minus1)
        start_token = self._token_id("start")
        end_token = self._token_id("end")
        current_token = start_token
        
        caption = []
        
        for _ in range(max_len):
            x_t = self.embedding.forward(current_token) 
            
            probs = self.forward_step(x_t)
            
            next_token = np.argmax(probs)
            
            next_word = self.text_util.idx_to_word.get(int(next_token), "")

            if next_token == end_token or next_word in ("", "end", "<end>"):
                break

            caption.append(next_word)
            current_token = next_token
        
        return ' '.join(caption)

    def generate_caption_from_image(
        self,
        image_path,
        keras_encoder,
        target_size=(299, 299),
        preprocess_fn=None,
        max_len=35,
    ):
        feature = self.extract_feature_from_image(
            image_path=image_path,
            keras_encoder=keras_encoder,
            target_size=target_size,
            preprocess_fn=preprocess_fn,
        )
        return self.generate_caption(feature, max_len=max_len)

    @staticmethod
    def extract_feature_from_image(image_path, keras_encoder, target_size=(299, 299), preprocess_fn=None):
        from PIL import Image

        image_path = Path(image_path)
        img = Image.open(image_path).convert("RGB").resize(target_size)
        arr = np.asarray(img, dtype=np.float32)
        batch = np.expand_dims(arr, axis=0)
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)
        else:
            batch = batch / 255.0
        feature = keras_encoder.predict(batch, verbose=0)
        return feature.reshape(feature.shape[0], -1)[0]

    def _token_id(self, token):
        token_id = self.text_util.word_to_idx.get(token)
        if token_id is None:
            token_id = self.text_util.word_to_idx.get(f"<{token}>")
        if token_id is None:
            raise KeyError(f"Token '{token}' tidak ada di vocabulary.")
        return token_id
