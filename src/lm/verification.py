

class Attributor():

    def __init__(self, models):
        """
        Constructor

        Parameters:
        ===========
            - dict `models`: a language model (value) 
              for each individual author.
        """
        self.models = models
        self.authors = tuple(sorted(models.keys()))

    def predict_probas(self, texts):
        """
        Parameters:
        ===========
            - an iterable of strings to be attributed

        Returns:
        ===========
            - a array with probability scores of size:
              nb texts x nb candidate authors
        """
        text_probas = []

        for text in texts:
            probas.append([models[author].predict_proba(text=text,
                                                        author=author)
                        for authors in self.authors])

        return np.array(text_probas, dtype=np.float32)

    def predict(self, texts):
        """
        Parameters:
        ===========
            - an iterable of strings to be attributed

        Returns:
        ===========
            - a list of strings with the author attribution
              for each text.
        """
        probas = self.get_probas(texts)
        return [self.authors[idx] for idx in probas.argmax(axis=-1)]
    

