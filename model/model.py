from utils.print_decorator import print_if_complete
from sklearn import linear_model, metrics
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import gensim
import xgboost as xgb


class Model():
    def __init__(self):
        self.model = self._create()

    def _create(self, config):
        if classification == "log":
            model = linear_model.LogisticRegression()
        elif classification == "svc":
            model = LinearSVC()
        elif classification == "knear":
            model = KNeighborsClassifier(n_neighbors=3)
        elif classification == "randomTree":
            model = RandomForestClassifier(n_estimators=100)
        elif classification == "xgb":
            model = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                                      subsample=0.8, nthread=10, learning_rate=0.1)
        return model

    @print_if_complete
    def train(self, X_train, y):
        self.model.fit(X_train, y)

    def get(self):
        return self.model

    @print_if_complete
    def save(self, name):
        print(self.model)
        filename = 'model/'+name+'_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    @print_if_complete
    def load(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def evaluate(self, X_test, y_test):
        print(self.model.score(X_test, y_test))
