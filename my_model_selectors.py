import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    # def select(self):
    #     raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_comp = None
        best_score = float('-inf')

        for components in range (self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(components).fit(self.X, self.lengths)

                logL = model.score(self.X, self.lengths)
                p = components*components + 2*components*len(self.X[0])-1
                N = len(self.X)

                bic = -2*logL + p*math.log(N)

                if bic > best_score:
                    best_score = bic
                    best_comp = components
            except:
                pass

        return self.base_model(best_comp)



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_comp = None
        best_score = float('-inf')

        for components in range (self.min_n_components, self.max_n_components+1):
            try:
                words_scores = {}
                model = self.base_model(components).fit(self.X, self.lengths)
                score = model.score(self.X, self.lengths)
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        word_score = model.score(X, lengths)
                        words_scores[word] = word_score

                if words_scores:
                    dic = (score - np.mean([words_scores[word] for word in words_scores.keys()]))

                    if dic > best_score:
                        best_score = dic
                        best_comp = components
            except:
                pass

        return self.base_model(best_comp)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_comp = None
        best_score = float('-inf')

        if len(self.lengths) > 1:
            num_splits = min(len(self.lengths), 3)
            split_method = KFold(num_splits)

        for components in range (self.min_n_components, self.max_n_components+1):
            try:
                score = 0
                total = 0
                model = None

                if len(self.lengths) > 1:
                    for cv_train, cv_test in split_method.split(self.sequences):
                        X_train, length_train = combine_sequences(cv_train, self.sequences)
                        X_test, length_test = combine_sequences(cv_test, self.sequences)

                        model = self.base_model(components).fit(X_train, length_train)
                        score += model.score(X_test, length_test)
                        total += 1

                        score = score/total
                else:
                    model = self.base_model(components).fit(self.X, self.lengths)
                    score += model.score(self.X, self.lengths)
                    total += 1

                    score = score/total

                if score > best_score:
                    best_score = score
                    best_comp = components
            except:
                pass

        return self.base_model(best_comp)