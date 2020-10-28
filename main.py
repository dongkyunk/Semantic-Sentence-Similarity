import warnings
import config
from utils.load_data import readData
from preprocess import preprocess, word_embedding
from model.model import Model
from feature_extraction.feature_extraction import feature_extraction

# Ignore Warning
warnings.filterwarnings(action='ignore')

# Read csv data
train_df, test_df, sample_df = readData()

xtrain1, xvalid1, xtrain2, xvalid2, y_train, y_valid = preprocess(train_df)
word_embedding = word_embedding(train_df, test_df)
x_train, x_valid, sims_test = feature_extraction(
    xtrain1, xtrain2, xvalid1, xvalid2, test_df)

# Model init and train
model = Model()
model.train(x_train, y_train)

# Evaluate on train
model.evaluate(x_train, y_train)

# Evaluate on validation
model.evaluate(x_valid, y_valid)
