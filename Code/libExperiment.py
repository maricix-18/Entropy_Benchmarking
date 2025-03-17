"""
Useful class for handling an experiment
"""

from libQC import BackendParams, CircuitParams, NoiseParams, ProtocolParams
from libShadows import CSParams
from libSWAP import SWAPParams

class ExperimentParams:
    def __init__(self, backend_params, circuit_params, noise_params, protocol_params, seed, timestamp=None, csv_files=dict(), metrics_file=""):
        self.backend_params = backend_params
        self.circuit_params = circuit_params
        self.noise_params = noise_params
        self.protocol_params = protocol_params
        self.seed = seed
        self.timestamp = timestamp
        self.csv_files = csv_files
        self.metrics_file = metrics_file
    def to_dict(self):
        temp = dict()
        temp['seed'] = self.seed
        temp['timestamp'] = self.timestamp
        temp['backend_params'] = vars(self.backend_params)
        temp['noise_params'] = vars(self.noise_params)
        temp['protocol_params'] = vars(self.protocol_params)
        temp['circuit_params'] = self.circuit_params.to_dict()
        temp['csv_files'] = self.csv_files
        temp['metrics_file'] = self.metrics_file
        return temp
    @staticmethod
    def from_dict(temp):
        backend_params = BackendParams.from_dict(temp['backend_params'])
        circuit_params = CircuitParams.from_dict(temp['circuit_params'])
        noise_params = NoiseParams.from_dict(temp['noise_params'])
        if temp['protocol_params']['name'] == 'CS':
            protocol_params = CSParams.from_dict(temp['protocol_params'])
        elif temp['protocol_params']['name'] == 'SWAP':
            protocol_params = SWAPParams.from_dict(temp['protocol_params'])
        elif temp['protocol_params']['name'] == 'DensMat':
            protocol_params = ProtocolParams('DensMat', temp['protocol_params']['num_samples'])
        return ExperimentParams(backend_params, circuit_params, noise_params, protocol_params, temp['seed'], temp['timestamp'], temp['csv_files'], temp['metrics_file'])
