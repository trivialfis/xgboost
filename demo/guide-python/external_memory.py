import os
import xgboost as xgb
import numpy as np

## simple example for using external memory version

# this is the only difference, add a # followed by a cache prefix name
# several cache file with the prefix will be generated
# currently only support convert from libsvm file
CURRENT_DIR = os.path.dirname(__file__)
dtrain = xgb.DMatrix(
    os.path.join(CURRENT_DIR, "../data/agaricus.txt.train?indexing_mode=1#dtrain.cache")
)
assert dtrain.num_col() == 126
dtest = xgb.DMatrix(
    os.path.join(CURRENT_DIR, "../data/agaricus.txt.test?indexing_mode=1#dtest.cache")
)
assert dtest.num_col() == 126

# performance notice: set nthread to be the number of your real cpu some cpu offer two
# threads per core, for example, a 4 core cpu with 8 threads, in such case set nthread=4
# param['nthread'] = num_real_cpu
param = {
    "max_depth": 2,
    "eta": 1,
    "objective": "binary:logistic",
    "tree_method": "approx",
}

# specify validations set to watch performance
watchlist = [(dtest, "eval"), (dtrain, "train")]
num_round = 2
from_ext = xgb.train(param, dtrain, num_round, watchlist)
predt_from_ext = from_ext.predict(dtrain)

dtrain = xgb.DMatrix(os.path.join(CURRENT_DIR, "../data/agaricus.txt.train"))
dtest = xgb.DMatrix(os.path.join(CURRENT_DIR, "../data/agaricus.txt.test"))

watchlist = [(dtest, "eval"), (dtrain, "train")]
num_round = 2
from_blob = xgb.train(param, dtrain, num_round, watchlist)

predt_from_blob = from_blob.predict(dtrain)

np.testing.assert_allclose(predt_from_ext, predt_from_blob)
