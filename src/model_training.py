import joblib
import comet_ml
import numpy as np
import os
import sys
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from src.logger import get_logger
from src.custom_exception import CustomException
from src.base_model import BaseModel
from utils.common_functions import read_yaml
from config.paths_config import *

logger=get_logger(__name__)

class ModelTraining:
    def __init__(self,data_path,config_path):
        self.data_path=data_path
        self.config=read_yaml(config_path)
        self.experiment=comet_ml.Experiment(
            api_key=self.config["API"]["comet_ml_API"],
            project_name="Anime_Recommender_Proj",
            workspace="anurag-dey"
        )
        logger.info("Model Training & COMET ML initialized")
        
    def load_data(self):
        try:
            X_train_array=joblib.load(X_TRAIN_ARRAY)
            X_test_array=joblib.load(X_TEST_ARRAY)
            y_train_array=joblib.load(y_TRAIN_ARRAY)
            y_test_array=joblib.load(y_TEST_ARRAY)
            
            logger.info("Data loaded succesfully for Model Training")
            return X_train_array,X_test_array,y_train_array,y_test_array
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_model(self):
        try:
            X_train_array,X_test_array,y_train_array,y_test_array=self.load_data()
            n_users=len(joblib.load(USER2USER_encoded))
            n_anime=len(joblib.load(ANIME2ANIME_decoded))
            
            
            base_model=BaseModel(config_path=CONFIG_PATH)
            
            model=base_model.RecommenderNet(n_users=n_users,n_anime=n_anime)
            start_lr=0.00001
            min_lr=0.00001
            max_lr=0.00005
            batch_size=10000

            ramup_epochs=5
            sustain_epochs=0
            exp_decay=0.8

            def lrfn(epoch):
                if epoch<ramup_epochs:
                    return (max_lr-start_lr)/ramup_epochs*epoch +start_lr
                elif epoch<ramup_epochs+sustain_epochs:
                    return max_lr
                else:
                    
                    return (max_lr-min_lr)*exp_decay**(epoch-ramup_epochs-sustain_epochs)+min_lr
            
            lr_callback=LearningRateScheduler(lambda epoch:lrfn(epoch),verbose=True)
            

            model_checkpoint_callback=ModelCheckpoint(
                filepath=CHECKPOINT_FILEPATH,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            early_stopping=EarlyStopping(
                patience=3,
                monitor='val_loss',
                mode='min',
                restore_best_weights=True)
            my_callbacks=[model_checkpoint_callback,lr_callback,early_stopping]
            
            os.makedirs(os.path.dirname(CHECKPOINT_FILEPATH),exist_ok=True)
            os.makedirs(MODEL_DIR,exist_ok=True)
            os.makedirs(WEIGHTS_DIR,exist_ok=True)
            
            try:
               history=model.fit(
                    x=X_train_array,
                    y=y_train_array,
                    batch_size=batch_size,
                    epochs=20,
                    validation_data=(X_test_array,y_test_array),
                    verbose=1,
                    callbacks=my_callbacks
                )
               model.load_weights(CHECKPOINT_FILEPATH)
               logger.info("Model Training Complete")
               
               for epoch in range(len(history.history['loss'])):
                   train_loss=history.history['loss'][epoch]
                   val_loss=history.history['val_loss'][epoch]
                   
                   self.experiment.log_metric('train_loss',train_loss,step=epoch)
                   self.experiment.log_metric('val_loss',val_loss,step=epoch)

               
            except Exception as e:
                raise CustomException(e,sys)
            self.save_model_weights(model)
        except Exception as e:
            logger.error(str(e))
            raise CustomException(e,sys)
        
    def extract_weights(self,layer_name,model):
        try:
            weight_layer=model.get_layer(layer_name)
            weights=weight_layer.get_weights()[0]
            weights=weights/np.linalg.norm(weights,axis=1).reshape((-1,1))
            logger.info(f"Extracting weights for {layer_name}")
            return weights
        except Exception as e:
            return CustomException(e,sys)
        
    def save_model_weights(self,model):
        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")
            
            user_weights=self.extract_weights('user_embeddding',model)
            anime_weights=self.extract_weights('anime_embeddings',model)
            
            joblib.dump(user_weights,USER_WEIGHTS_PATH)
            joblib.dump(anime_weights,ANIME_WEIGHTS_PATH)
            
            self.experiment.log_asset(MODEL_PATH)
            self.experiment.log_asset(ANIME_WEIGHTS_PATH)
            self.experiment.log_asset(USER_WEIGHTS_PATH)
            logger.info("USer and Anime weights saved successfully")
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__=="__main__":
    
    model_trainer=ModelTraining(PROCESSED_DIR,CONFIG_PATH)
    model_trainer.train_model()  