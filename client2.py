import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import flwr as fl

#Declaration of certain variables
eval_accuracy = []
eval_loss = []

#Loading the CSV data file for model training                          
df = pd.read_csv('dataset_abnormal_client2.csv')

#setting up the global seed to control the randomness                                                     
tf.random.set_seed(42)

# DATA PREPROCESSING

# Remove unused features
df_removed = df.drop(columns=['time', 'yaw', 'heading', 'location_x', 'location_y', 'gnss_latitude', 'gnss_longitude', 'gyroscope_x', 'gyroscope_y', 'height', 'reverse', 'hand_brake', 'manual_gear_shift', 'gear'])

# Engineer new features
import feature_engineering as fteng
df_engineered = fteng.engineer_features(df_removed)

# Aggregate all features
import feature_aggregration as ftagg
df_aggregated = ftagg.aggregate_features(df_engineered)

# Define input features and target
X = df_aggregated[['mean_speed', 'max_speed', 'mean_acceleration', 'max_acceleration', 'mean_yaw_speed', 'max_yaw_speed', 'mean_throttle', 'max_throttle_count', 'mean_steer_change', 'max_steer_change', 'mean_positive_brake_change', 'max_positive_brake_change']]
Y = df_aggregated['abnormality']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.8)

# Scale input features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X1_row, X1_col = np.shape(X_train)

#Building the model
dbp2_model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(X1_col,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])														

dbp2_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])	

keras.backend.clear_session()

#Explicitly setting the initial weights to decrease randomness of output
initial_weights_filename = "initial_weights_client2_1.h5"
dbp2_model.load_weights(initial_weights_filename)

#Loading the test dataset
#Here client1 training dataset is used as testing dataset for client2
df_test = pd.read_csv('dataset_normal_client1.csv')

df_test_removed = df_test.drop(columns=['time', 'yaw', 'heading', 'location_x', 'location_y', 'gnss_latitude', 'gnss_longitude', 'gyroscope_x', 'gyroscope_y', 'height', 'reverse', 'hand_brake', 'manual_gear_shift', 'gear'])

# Engineer new features
df_test_engineered = fteng.engineer_features(df_test_removed)

# Aggregate all features
import feature_aggregration as ftagg
df_test_aggregated = ftagg.aggregate_features(df_test_engineered)

X_test = df_test_aggregated[['mean_speed', 'max_speed', 'mean_acceleration', 'max_acceleration', 'mean_yaw_speed', 'max_yaw_speed', 'mean_throttle', 'max_throttle_count', 'mean_steer_change', 'max_steer_change', 'mean_positive_brake_change', 'max_positive_brake_change']]
y_test = df_test_aggregated['abnormality']

X_test = scaler.transform(X_test)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return dbp2_model.get_weights()

    def fit(self, parameters, config):
    
        dbp2_model.set_weights(parameters)
        
        r1 = dbp2_model.fit(X_train, y_train, batch_size= len(X_train), epochs = 5, validation_data = (X_valid, y_valid), verbose=0)
        hist1 = r1.history
        print("Fit history : " , hist1)
        #print("weights of client1: ", dbp2_model.get_weights())
        #print("Size of weights in client1 in bytes: ", sys.getsizeof(dbp2_model.get_weights()))
        #print(type(dbp2_model.get_weights()))
        
        return dbp2_model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
    
        #The evaluate function receives aggregated parameters from the server and we set those parameters and evaluate the model        
        dbp2_model.set_weights(parameters)
        
        loss, accuracy = dbp2_model.evaluate(X_test, y_test, verbose=0)
        
        #Saving the evaluation values for plotting                             
        eval_accuracy.append(accuracy)
        
        #Saving the loss values for plotting                      
        eval_loss.append(loss)
        #print("Eval accuracy : ", accuracy)
        
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address = "localhost: 8080", 
        client = FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)

del dbp2_model