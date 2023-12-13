#Importing necessary libraries
import flwr as fl
import numpy as np
import encryption as encr
import tenseal as ts
import filedata as fd


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, min_fit_clients, min_available_clients):
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.avg_accuracy = 0.0
        
        #The server calls create_context function from encryption.py file 
        #The code is developed in a way that the server generates the secret key and public key since it is the starting point and synchronises the communication
        if encr.Enc_needed.encryption_needed.value:
            encr.create_context()

    def aggregate_fit(self, rnd, results, failures):
        
        if len(results) < self.min_available_clients:
            print(f"Not enough clients available (have {len(results)}, need {self.min_available_clients}). Skipping round {rnd}.")
            return None
        
        #Loading public key to perform computations on encrypted data    
        public_key_context = ts.context_from(fd.read_data("public_key.txt")) 
        
        if encr.Enc_needed.encryption_needed.value == 1:                                    #Full encryption is selected
            
            #Declaration of array to store aggregated parameters
            aggregated_weights = []
            
            #Loading encrytped vector list of client - 1
            inp1_proto_ex = fd.read_data('data_encrypted_Client1.txt')
            inp1_ex = ts.lazy_ckks_tensor_from(inp1_proto_ex)
            inp1_ex.link_context(public_key_context)
            
            #Loading encrypted vector list of client - 2
            inp2_proto_ex = fd.read_data('data_encrypted_Client2.txt')
            inp2_ex = ts.lazy_ckks_tensor_from(inp2_proto_ex)
            inp2_ex.link_context(public_key_context)
            
            #Adding the parameters
            results1_ex = (inp1_ex) + (inp2_ex)
            
            #Dividing with number of clients -> Averaging
            denominator_plain_ex = ts.plain_tensor([0.5])
            denominator_ckks_ex = ts.ckks_tensor(public_key_context, denominator_plain_ex)
            results_ex = results1_ex * denominator_ckks_ex
            
            #Storing the aggregated result in a file
            result_ex_file_path = 'result_ex.txt'
            fd.write_data(result_ex_file_path, results_ex.serialize())
            
            #As Flower framework does not CKKS encrypted objects, aggregation is by-passed with user-defined function (see above computations)
            #In order to continue simulation, aggregation is performed here with in-built functions
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            
        elif encr.Enc_needed.encryption_needed.value == 2:                                    #Partial encryption is selected
            
            #Declaration of array to store aggregated parameters
            aggregated_weights = []
            
            #Loading encrytped vector list of client - 1
            inp1_proto_ex = fd.read_data('data_encrypted_2_Client1.txt')
            inp1_ex = ts.lazy_ckks_tensor_from(inp1_proto_ex)
            inp1_ex.link_context(public_key_context)
            
            #Loading encrypted vector list of client - 2
            inp2_proto_ex = fd.read_data('data_encrypted_2_Client2.txt')
            inp2_ex = ts.lazy_ckks_tensor_from(inp2_proto_ex)
            inp2_ex.link_context(public_key_context)
            
            #Adding the parameters
            results2_ex = (inp1_ex) + (inp2_ex)
            
            #Dividing with number of clients -> Averaging
            denominator_plain_ex = ts.plain_tensor([0.5])
            denominator_ckks_ex = ts.ckks_tensor(public_key_context, denominator_plain_ex)
            results_ex = results2_ex * denominator_ckks_ex
            
            #Storing the aggregated result in a file
            result_ex_file_path = 'result_ex_2.txt'
            fd.write_data(result_ex_file_path, results_ex.serialize())
            
            #As Flower framework does not CKKS encrypted objects, aggregation is by-passed with user-defined function (see above computations)
            #In order to continue simulation, aggregation is performed here with in-built functions
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            
        else:                                                                               #No encryption is selected
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
           
        return aggregated_weights
    
# Create strategy and run server
min_fit_clients = 2
min_available_clients = 2
strategy = SaveModelStrategy(min_fit_clients, min_available_clients)

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='localhost:8080',
    config=fl.server.ServerConfig(num_rounds = 10),
    grpc_max_message_length=1024*1024*1024,
    strategy=strategy
)

