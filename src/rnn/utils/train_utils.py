import numpy as np

class DataGenerator:
    def __init__(self, mapping_data, image_features, vocab_size=None, sequence_length=35, batch_size=32):
        self.mapping_data = mapping_data 
        self.image_features = image_features 
        self.img_keys = list(mapping_data.keys())
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.n_images = len(self.img_keys)

    def generate(self):
        while True:
            np.random.shuffle(self.img_keys)
            
            for i in range(0, self.n_images, self.batch_size):
                batch_keys = self.img_keys[i:i+self.batch_size]
                
                X_img, X_txt, y_label = [], [], []
                
                for img_name in batch_keys:
                    feature = self.image_features[img_name]
                    sequences = self.mapping_data[img_name]
                    
                    for seq in sequences:
                        input_seq = seq[:-1]
                        target_seq = seq[1:]
                        
                        X_img.append(feature)
                        X_txt.append(input_seq)
                        y_label.append(target_seq)
                
                yield (np.array(X_img), np.array(X_txt)), np.array(y_label)
