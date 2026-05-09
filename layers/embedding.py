class EmbeddingLayer:
    def __init__(self):
        self.W_e = None
    
    def set_w(self, weights:list[list]):
        self.W_e = weights

    def forward(self, words: list):
        if self.W_e is None:
            raise ValueError("Bobot belum di-load!")
        
        return self.W_e[words]