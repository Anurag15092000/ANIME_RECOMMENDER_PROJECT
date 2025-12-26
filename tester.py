from utils.helpers import *
from config.paths_config import *
from pipeline.prediction_pipeline import hybrid_recommendation

print(USER_WEIGHTS_PATH)

print(hybrid_recommendation(11880))