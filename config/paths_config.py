import os



###################### DATA INGESTION #########################

RAW_DIR="artifacts/raw"
CONFIG_PATH="config/config.yaml"

###################### DATA PROCESSING ########################

PROCESSED_DIR="artifacts/processed"
ANIMELIST_CSV="artifacts/raw/animelist.csv"
ANIME_CSV="artifacts/raw/anime.csv"
SYNOPSIS_CSV="artifacts/raw/anime_with_synopsis.csv"

X_TRAIN_ARRAY=os.path.join(PROCESSED_DIR,"X_train_array.pkl")
X_TEST_ARRAY=os.path.join(PROCESSED_DIR,"X_test_array.pkl")
y_TRAIN_ARRAY=os.path.join(PROCESSED_DIR,"y_train_array.pkl")
y_TEST_ARRAY=os.path.join(PROCESSED_DIR,"y_test_array.pkl")


RATING_DF=os.path.join(PROCESSED_DIR,"rating_df.csv")
DF=os.path.join(PROCESSED_DIR,"anime_df.csv")
SYNOPSIS_DF=os.path.join(PROCESSED_DIR,"synopsis_df.csv")

USER2USER_encoded=os.path.join(PROCESSED_DIR,"user2user_encoded.pkl")
USER2USER_decoded=os.path.join(PROCESSED_DIR,"user2user_decoded.pkl")
ANIME2ANIME_encoded=os.path.join(PROCESSED_DIR,"anime2anime_encoded.pkl")
ANIME2ANIME_decoded=os.path.join(PROCESSED_DIR,"anime2anime_decoded.pkl")


MODEL_DIR="artifacts/model"
WEIGHTS_DIR="artifacts/weights"
MODEL_PATH=os.path.join(MODEL_DIR,"model.h5")
USER_WEIGHTS_PATH=os.path.join(WEIGHTS_DIR,"anime_weights.pkl")
ANIME_WEIGHTS_PATH=os.path.join(WEIGHTS_DIR,"user_weights.pkl")
CHECKPOINT_FILEPATH='artifacts/model_checkpoint/weights.weights.h5'