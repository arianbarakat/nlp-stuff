

class word2vecPreprocessor():

    def __init__(self, text_input):
        self.text_input = text_input

    def _tokenize(self):
        from nltk.tokenize import sent_tokenize, word_tokenize

        sentences = []
        for text_input in self.text_input:
            for  possible_sent in  sent_tokenize(text_input):
                sentences.append(possible_sent)

        self.sentences = sentences

        self.tokens = [word_tokenize(sentence) for  sentence in sentences]

    def _context_builder(self, tokens, wdw_size, model_type):

        assert isinstance(model_type, str)
        assert model_type.lower() in ["cbow", "skip-gram"]

        contexts =  [] # ([...,contex_token_i-1,contex_token_i,...,], target_token)


        for i, token in enumerate(tokens):
            left = 0 if i - wdw_size < 0 else i - wdw_size
            right = len(tokens) if i + wdw_size + 1  > len(tokens) else i + wdw_size + 1

            left_context =  tokens[left:i]
            right_contex = tokens[i+1:right]
            contexts.append((left_context + right_contex, tokens[i]))

        if model_type.lower() == "skip-gram":
            _contex_skipgram = []

            for _context_token in contexts:
                for _contex_single_tuple in list(zip(_context_token[0], [_context_token[1]]*len(_context_token[0]))):
                    _contex_skipgram.append(_contex_single_tuple)

            return _contex_skipgram

        if model_type.lower() == "cbow":
            _context_cbow = contexts

            return _context_cbow


    def _to_one_hot_encoding(self, context_tuple):
        import numpy as np

        assert hasattr(self,"vocabSize")
        assert hasattr(self,"token2int")
        assert hasattr(self,"int2token")

        _contex_tokens, _token =  context_tuple

        if isinstance(_contex_tokens, str):
            _contex_tokens = [_contex_tokens]

        token_one_hot_encoding = np.zeros(self.vocabSize)
        context_one_hot_encoding = np.zeros(self.vocabSize)

        token_one_hot_encoding[self.token2int[_token]] = 1

        for context_token in _contex_tokens:
             context_one_hot_encoding[self.token2int[context_token]] = 1

        return context_one_hot_encoding, token_one_hot_encoding


    def _vocabulary_builder(self):
        _token2int = {}
        _int2token = []

        assert hasattr(self, "tokens")

        for sent_tokens in self.tokens:
            for token in sent_tokens:
                if not token in _token2int.keys():
                    idx = len(_int2token)
                    _int2token.append(token)
                    _token2int[token] =  idx

        self.vocab = _token2int.keys()
        self.vocabSize = len(_int2token)
        self.token2int = _token2int
        self.int2token = _int2token


    def get_word2vec_data(self, context_size, model_type):
        import numpy as np
        self._tokenize()
        self._vocabulary_builder()

        assert isinstance(context_size, int)
        assert isinstance(model_type, str)
        assert model_type.lower() in ["cbow", "skip-gram"]

        _token = []
        _context = []

        self.contexts = []
        n = len(self.tokens)

        for i, sent_tokens in enumerate(self.tokens):
            for context_tuple in self._context_builder(sent_tokens, wdw_size=context_size, model_type=model_type):
                self.contexts.append(context_tuple)
                _contex_obs,  _token_obs = self._to_one_hot_encoding(context_tuple=context_tuple)
                _token.append(_token_obs)
                _context.append(_contex_obs)

        if model_type == "cbow":
            _target = _token
            _input = _context

        if model_type == "skip-gram":
            _target = _context
            _input = _token

        return np.array(_target), np.array(_input)
