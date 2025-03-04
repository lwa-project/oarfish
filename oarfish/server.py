import zmq
import json
import queue
import numpy as np
from logging import Logger
from functools import lru_cache
from typing import Optional, Any, Tuple, Union

from torch.utils.data import DataLoader

from astropy.coordinates import EarthLocation

from .data import SingleChannelDataset, MultiChannelDataset

class PredictionServer:
    def __init__(self, address: str='127.0.0.1', port: int=5555, timeout: float=1.0,
                 predictor: Optional[Any]=None, logger: Optional[Logger]=None):
        self.address = address
        self.port = port
        self.timeout = timeout
        self.predictor = predictor
        self.logger = logger
        
    def start(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, int(self.timeout*1000))
        
        self.sock.bind(f"tcp://{self.address}:{self.port}")
        
        if self.logger:
            self.logger.info(f"Started prediction server on {self.address} port {self.port}")
            
    def end(self):
        self.sock.close()
        self.ctx.term()
        
        if self.logger:
            self.logger.info(f"Stopped prediction server on {self.address} port {self.port}")
            
    def receive(self) -> bool:
        try:
            parts = self.sock.recv_multipart()
            
        except zmq.error.Again:
            return False
            
        try:
            if len(parts) == 4:
                results = self.process(*parts)
            else:
                results = self.identify(*parts)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to predict request from {parts[0]}, request ID {parts[1]}: {str(e)}")
            return False
            
        try:
            self.sock.send_multipart(results)
        except zmq.error.Again as e:
            if self.logger:
                self.logger.warn(f"Failed to send results on {results[1]} to {results[0]}: {str(e)}")
            return False
            
        return True
    
    def identify(self, client_id: bytes, request_id: bytes) -> Tuple[bytes, bytes, bytes]:
        ident = self.predictor.identify()
        return client_id, request_id, json.dumps(ident).encode()
        
    @staticmethod
    @lru_cache(maxsize=8)
    def _get_station_location(lon: str, lat: str, height: Union[str,float]) -> EarthLocation:
        return EarthLocation(lon=lon,
                             lat=lat,
                             height=height)
        
    def process(self, client_id: bytes, request_id: bytes, metadata: bytes, image_cube: bytes) -> Tuple[bytes, bytes, bytes]:
        metadata = json.loads(metadata)
        image_cube = np.frombuffer(image_cube, dtype=np.float32)
        image_cube = image_cube.reshape(*metadata['image_cube_shape'])
        image_cube = image_cube.copy()
        
        location = None
        if 'lon' in metadata and 'lat' in metadata:
            location = self._get_station_location(metadata['lon'],
                                                  metadata['lat'],
                                                  metadata['height'] if 'height' in metadata else 0.0)
            
        dataset = MultiChannelDataset(metadata,
                                      image_cube[:,0,:,:],
                                      image_cube[:,-1,:,:],
                                      station_location=location)
        
        results = self.predictor.predict_dataset(dataset)
        
        return client_id, request_id, json.dumps(results).encode()
