import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import flwr as fl
import sys
import matplotlib.pyplot as plt
import encryption as encr
import aggregator as aggr


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
        
        r2 = dbp2_model.fit(X_train, y_train, batch_size= len(X_train), epochs = 5, validation_data = (X_valid, y_valid), verbose=0)
        hist2 = r2.history
        print("Fit history : " , hist2)
        #print("weights of client2: ", dbp2_model.get_weights())
        #print("Size of weights in client1 in bytes: ", sys.getsizeof(dbp2_model.get_weights()))
        #print(type(dbp2_model.get_weights()))
        
        #Adding encryption layer for dbp2_model.get_weights() and then send them for aggregation
        
        #Checking if the encryption is necessary based on the boolean variable 'encryption_needed' from encryption.py file
        if encr.Enc_needed.encryption_needed.value:
        
            params_encrypted = []
            
            print("Entering encryption phase - Client2")
            params_encrypted_list = encr.param_encrypt(dbp2_model.get_weights(), 'Client2')
            print("Exiting encryption phase - Client2")
            
            #Appending the encrypted data to a list. Here a loop is introduced to access the elements from list variable 'param_encrypted_list'
            for i in range(0, 514):
                params_encrypted.append(params_encrypted_list[i])
            
            print("Entering aggregation phase - Client1")
            global weights
            
            #The encrypted weights undergo aggregation in aggregator.py file and returns the new aggregated weights
            weights = aggr.user_aggregator()
            #print('weights in fit method: ', weights)
            print("Exiting aggregation phase - Client2")
            
        else:
            
            params_encrypted = dbp2_model.get_weights()    
                 
        return dbp2_model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        
        #Checking if the decryption is necessary based on the boolean variable 'encryption_needed' from encryption.py file
        if encr.Enc_needed.encryption_needed.value:
        
            print("Entering decryption phase - Client2") 
            params_decrypted2 = encr.param_decrypt(weights, 'Client2')
            print("Exiting decryption phase - Client2")
            
            #The parameters are needed to be reshaped back to original as they are flattened during encryption and aggregation
            
            # List to store the reshaped arrays
            reshaped_params = []

            # Define the shapes of the original arrays
            shapes = [np.shape(arr) for arr in parameters]

            # Variable to keep track of the current index in the data
            current_index = 0

            # Reshape the data and split it into individual arrays
            for shape in shapes:
                data_result = []
                size = np.prod(shape)
                                    
                reshaped_arr = np.reshape(params_decrypted2[current_index:current_index + size], shape)
                #print(reshaped_arr)
                reshaped_params.append(reshaped_arr)
                current_index += size

            #print('reshaped params of client2: ', reshaped_params)
            
            #Using the reshaped aggregated parameters to evaluate model performance
            dbp2_model.set_weights(reshaped_params)
            
        else:
            
            dbp2_model.set_weights(parameters)
        
        loss, accuracy = dbp2_model.evaluate(X_test, y_test, verbose=0)
        
        #Saving the evaluation values for plotting
        eval_accuracy.append(accuracy)
        
        #Saving the loss values for plotting
        eval_loss.append(loss)
        
        print("Eval accuracy : ", accuracy)
        
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address = "localhost: 8080", 
        client = FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)

#code snippet for confusion matrix
y1_pred = dbp2_model.predict(X_test)
#print(y1_pred)

y1_pred_labels = (y1_pred[:, 1] >= 0.51).astype(int)
confusion_client2 = confusion_matrix(y_test, y1_pred_labels)
print('Confusion Matrix:')
print(confusion_client2)

#plotting the model performance
x_points = np.array(range(1, 11))
print('Client2 - eval accuracy list:', eval_accuracy)
y_points1 = np.array(eval_accuracy)
y_points2 = np.array(eval_loss)

plt.plot(x_points, y_points1, color = 'green', label = 'Eval Accuracy')
plt.plot(x_points, y_points2, color = 'blue', label = 'Eval Loss')
plt.legend()
plt.title("Model Evaluation Metrics plot - client 2")
plt.xlabel("Number of FL rounds -->")
plt.ylabel("Accuracy and Loss metrics of the Model -->")
plt.show()

del dbp2_model