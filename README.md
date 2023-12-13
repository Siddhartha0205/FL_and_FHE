**Federated Learning** is a technique that allows training the machine learning or deep learning models over distributed nodes without the need to share data. This collaborative learning approach is introduced to mitigate the risk of privacy issues observed in traditional machine learning. The users hold a part of the entire distributed data without providing access to the server. Only model updates from the client site upon training the shared model with locally available data are shared with the server for aggregation. The aggregated weights are shared back to the clients for further training. 

Over a period of time, several attacks against the FL framework with the intent to leak private information by capturing model updates shared between the server and clients are identified. With technological advancements, the adversaries could successfully reconstruct private data by analyzing these model updates. Consequently, users’ confidentiality is compromised. Hence, an additional privacy scheme is necessary to mitigate the risk of such confidentiality-compromising attacks.

**Fully Homomorphic Encryption**, a privacy-preserving scheme is introduced to encrypt the model updates at the client site. This scheme supports performing arithmetic computations on encrypted data without any need for decryption providing the same results as computations performed on unencrypted data. Though it provides privacy to a great extent, the scheme has a downside that it introduces a huge computational overhead. Hence, the encryption scheme is optimized in this project and the results are compared. 

server.py - contains the code necessary to define the strategy for orchestrating the FL scenario
client1.py and client2.py - contain the code necessary for a client to participate in FL setup
encryption.py - contains the code necessary for encryption and decryption and information about the depth of encryption
filedata.py - contains the code to serialize and de-serialize the data

Because of the constraint of the size of the dataset, the training dataset of client1 is used as the test dataset for client2 and the training dataset of client2 is used as the test dataset for client1

**Limitations of the project**: Flower protocol is bypassed as the framework doesn't support TenSEAL objects. The encrypted parameters are not shared using Flower protocol but written and read using text files. For the sake of completeness, objects obtained from the neural network are shared in the Flower protocol.
