import numpy as np

class DataGenerator:
    def __init__(self, mapping_data, image_features, vocab_size=None, sequence_length=35, batch_size=32, pad_id=0):
        self.mapping_data = mapping_data 
        self.image_features = image_features 
        self.img_keys = list(mapping_data.keys())
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.n_images = len(self.img_keys)
        self.pad_id = pad_id

    def generate(self):
        while True:
            np.random.shuffle(self.img_keys)
            
            for i in range(0, self.n_images, self.batch_size):
                batch_keys = self.img_keys[i:i+self.batch_size]
                
                X_img, X_txt, y_label, sample_weights = [], [], [], []
                
                for img_name in batch_keys:
                    feature = self.image_features[img_name]
                    sequences = self.mapping_data[img_name]
                    
                    for seq in sequences:
                        input_seq = seq[:-1]
                        target_seq = seq[1:]
                        
                        X_img.append(feature)
                        X_txt.append(input_seq)
                        y_label.append(target_seq)
                        sample_weights.append((target_seq != self.pad_id).astype(np.float32))
                
                yield (
                    np.asarray(X_img, dtype=np.float32),
                    np.asarray(X_txt, dtype=np.int32),
                ), np.asarray(y_label, dtype=np.int32), np.asarray(sample_weights, dtype=np.float32)


def make_caption_dataset(mapping_data, image_features, sequence_length=35, batch_size=32, pad_id=0):
    import tensorflow as tf

    steps = int(sequence_length) - 1
    generator = DataGenerator(
        mapping_data=mapping_data,
        image_features=image_features,
        sequence_length=sequence_length,
        batch_size=batch_size,
        pad_id=pad_id,
    ).generate

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),
                tf.TensorSpec(shape=(None, steps), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(None, steps), dtype=tf.int32),
            tf.TensorSpec(shape=(None, steps), dtype=tf.float32),
        ),
    )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)
