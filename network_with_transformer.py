class NetworkWithTransformer:
    def __init__(self, model, transformer):
        self.model = model
        self.transformer = transformer

    def __call__(self, input):
        input = self.transformer.transform(input)
        return self.model(input)

    def fit(self, **kwargs):
        return self.model.fit(**kwargs)

    def save(self, path):
        self.model.save(path)
