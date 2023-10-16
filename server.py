import flwr as fl
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, min_fit_clients, min_available_clients):
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        

    def aggregate_fit(self, rnd, results, failures):
        
        if len(results) < self.min_available_clients:
            print(f"Not enough clients available (have {len(results)}, need {self.min_available_clients}). Skipping round {rnd}.")
            return None
        
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

