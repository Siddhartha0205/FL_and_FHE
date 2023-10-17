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
df = pd.read_csv('dataset_normal_client1.csv')

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
dbp1_model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(X1_col,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])														

dbp1_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])	

keras.backend.clear_session()

#Explicitly setting the initial weights to decrease randomness of output
initial_weights_filename = "initial_weights_client1_1.h5"
dbp1_model.load_weights(initial_weights_filename)

#Loading the test dataset
df_test = pd.read_csv('dataset_abnormal_client2.csv')

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
        return dbp1_model.get_weights()

    def fit(self, parameters, config):
        
        dbp1_model.set_weights(parameters)
        
        r1 = dbp1_model.fit(X_train, y_train, batch_size= len(X_train), epochs = 5, validation_data = (X_valid, y_valid), verbose=0)
        hist1 = r1.history
        print("Fit history : " , hist1)
        #print("weights of client1: ", dbp1_model.get_weights())
        #print("Size of weights in client1 in bytes: ", sys.getsizeof(dbp1_model.get_weights()))
        #print(type(dbp1_model.get_weights()))
        
        #Storing the shape of weights arrays of the model
        #for arr in dbp1_model.get_weights():
        #    print("The shape of model weights: ", np.shape(arr))
        
        #print("2 - The shape of model weights: ", len(dbp1_model.get_weights()))
        
        #Adding encryption layer for dbp1_model.get_weights() and then send them for aggregation
        
        #Checking if the encryption is necessary based on the boolean variable 'encryption_needed' from encryption.py file
        if encr.Enc_needed.encryption_needed.value:
        
            params_encrypted = []
            
            print("Entering encryption phase - Client1")
            params_encrypted_list_1 = encr.param_encrypt(dbp1_model.get_weights(), 'Client1')
            print("Exiting encryption phase - Client1")
            
            #Appending the encrypted data to a list. Here a loop is introduced to access the elements from list variable 'param_encrypted_list_1'  
            for i in range(0, 514):
                params_encrypted.append(params_encrypted_list_1[i])
                
            print("Entering aggregation phase - Client1")
            global weights1
            
            #The encrypted weights undergo aggregation in aggregator.py file and returns the new aggregated weights 
            weights1 = aggr.user_aggregator()
            #print('weights in fit method: ', weights1)
            print("Exiting aggregation phase - Client1")
            
        else:
            
            params_encrypted = dbp1_model.get_weights()    
            
        return dbp1_model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        
        #Checking if the decryption is necessary based on the boolean variable 'encryption_needed' from encryption.py file
        if encr.Enc_needed.encryption_needed.value:
                        
            print("Entering decryption phase - Client1")            
            params_decrypted1 = encr.param_decrypt(weights1, 'Client1')
            print("Exiting decryption phase - Client 1")
            
            #The parameters are needed to be reshaped to original as they are flattened during encryption and aggregation
            
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
                                    
                reshaped_arr = np.reshape(params_decrypted1[current_index:current_index + size], shape)
                #print(reshaped_arr)
                reshaped_params.append(reshaped_arr)
                current_index += size

            #print('reshaped params of client1: ', reshaped_params)
            
            #Using the reshaped aggregated parameters to evaluate model performance
            dbp1_model.set_weights(reshaped_params)
            
        else:
            
            dbp1_model.set_weights(parameters)
        
        loss, accuracy = dbp1_model.evaluate(X_test, y_test, verbose=0)
        
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

#code snippet for confusion matrix
y1_pred = dbp1_model.predict(X_test)
#print(y1_pred)

y1_pred_labels = (y1_pred[:, 1] >= 0.51).astype(int)
confusion_client1 = confusion_matrix(y_test, y1_pred_labels)
print('Confusion Matrix:')
print(confusion_client1)

#plotting the model performance
x_points = np.array(range(1, len(np.array(eval_accuracy))+1))
print('Client1 - eval accuracy list:', eval_accuracy)
y_points1 = np.array(eval_accuracy)
y_points2 = np.array(eval_loss)

plt.plot(x_points, y_points1, color = 'green', label = 'Eval Accuracy')
plt.plot(x_points, y_points2, color = 'blue', label = 'Eval Loss')
plt.legend()
plt.title("Model Evaluation Metrics plot - client 1")
plt.xlabel("Number of FL rounds -->")
plt.ylabel("Accuracy and Loss metrics of the Model -->")
plt.show()

del dbp1_model