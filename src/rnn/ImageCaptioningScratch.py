import numpy as np
from layers import EmbeddingLayer, DenseOutputLayer, DenseProjectionLayer, LSTMLayer

class ImageCaptioningModel:
    def __init__(self, keras_model, text_util, is_lstm=True):
        self.text_util = text_util
        self.is_lstm = is_lstm
        
        self.embedding = EmbeddingLayer()
        self.dense_projection = DenseProjectionLayer(2048, 256)
        self.lstm_layers = [] 
        self.dense_output = DenseOutputLayer(256, text_util.vocab_size)
        
        self._load_weights(keras_model)
    
    def _load_weights(self, keras_model):
        for layer in keras_model.layers:
            if 'embedding' in layer.name:
                emb_weights = layer.get_weights()[0]
                self.embedding.set_w(emb_weights)
                break

        for layer in keras_model.layers:
            if 'dense' in layer.name and 'Output_Layer' not in layer.name:
                proj_weights = layer.get_weights()
                self.dense_projection.set_weights(proj_weights[0], proj_weights[1])
                break
        
        layer_idx = 1
        while True:
            layer_name = f"LSTM_Layer_{layer_idx}" if self.is_lstm else f"RNN_Layer_{layer_idx}"
            try:
                layer = keras_model.get_layer(layer_name)
                weights = layer.get_weights()
                
                lstm = LSTMLayer()
                lstm.set_w(
                    units=layer.units,
                    kernel=weights[0],
                    recurrent_kernel=weights[1],
                    bias=weights[2]
                )
                self.lstm_layers.append(lstm)
                layer_idx += 1
            except:
                break
        
        out_weights = keras_model.get_layer('Output_Layer').get_weights()
        self.dense_output.set_weights(out_weights[0], out_weights[1])
    
    def reset_states(self):
        for lstm in self.lstm_layers:
            lstm.ht = np.zeros(lstm.units)
            lstm.ct = np.zeros(lstm.units)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def forward_step(self, input_t):
        x = input_t
        for lstm in self.lstm_layers:
            x = lstm.forward(x)
        logits = self.dense_output.forward(x.reshape(1, -1))
        probs = self.softmax(logits[0])
        return probs
    
    def generate_caption(self, image_feature, max_len=35):
        self.reset_states()
        
        x_minus1 = self.dense_projection.forward(
            image_feature.reshape(1, -1)
        )[0]
        
        self.forward_step(x_minus1)
        
        start_token = self.text_util.word_to_index['<start>']
        current_token = start_token
        
        caption = []
        
        for _ in range(max_len):
            x_t = self.embedding.forward(current_token) 
            
            probs = self.forward_step(x_t)
            
            next_token = np.argmax(probs)
            
            if next_token == self.text_util.word_to_index['<end>']:
                break
            
            caption.append(self.text_util.index_to_word[next_token])
            current_token = next_token
        
        return ' '.join(caption)