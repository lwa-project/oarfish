import zmq
import json
import time
import queue
import numpy as np
from logging import Logger
from functools import lru_cache
from collections import deque

from typing import Optional, Any, Tuple, Union, Dict

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
        
        self.request_stats = {'start_time': -1.0,
                              'total': 0,
                              'send_failed': 0,
                              'error': 0,
                              'success': 0,
                              'timeout': 0,
                              'last_success': -1.0,
                              'response_times': deque([], maxlen=100)
                             }
        
    def _reset_stats(self):
        """
        Reset the connection statistics to a clean state.
        """
        
        for key in ('start_time', 'last_success'):
            self.request_stats[key] = -1.0
        for key in ('total', 'send_failed', 'error', 'success', 'timeout'):
            self.request_stats[key] = 0
        self.request_stats['response_times'].clear()
        
    def start(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, int(self.timeout*1000))
        
        self._reset_stats()
        self.request_stats['start_time'] = time.time()
        
        self.sock.bind(f"tcp://{self.address}:{self.port}")
        
        if self.logger:
            self.logger.info(f"Started prediction server on {self.address} port {self.port}")
            
    def end(self):
        self.sock.close()
        self.ctx.term()
        
        if self.logger:
            self.logger.info(f"Stopped prediction server on {self.address} port {self.port}")
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics about the client that include:
         * how long it has been since start() was called
         * how many requests have been processed
         * when the last requests was successfully processed
         * the average processing time of the last 100 requests
        """
        
        uptime = 0.0
        if self.request_stats['start_time'] > 0:
            uptime = time.time() - self.request_stats['start_time']
            
        resp_time = 0.0
        if self.request_stats['response_times']:
            resp_time = sum(self.request_stats['response_times']) \
                        / len(self.request_stats['response_times'])
        
        last_good = None
        if self.request_stats['last_success'] > 0:
            last_good = self.request_stats['last_success']
            
        return {'running': self.ctx is not None,
                'uptime': uptime,
                'requests': {'total': self.request_stats['total'],
                             'send_failed': self.request_stats['send_failed'],
                             'error': self.request_stats['error'],
                             'timeout': self.request_stats['timeout'],
                             'successful': self.request_stats['success']
                            },
                'last_successful_response': last_good,
                'average_response_time': resp_time
               }
        
    def receive(self) -> bool:
        try:
            parts = self.sock.recv_multipart()
            
            t_start = time.time()
            self.request_stats['total'] += 1
            
        except zmq.error.Again:
            return False
            
        try:
            if len(parts) == 4:
                results = self.process(*parts)
            else:
                results = self.identify(*parts)
        except Exception as e:
            self.request_stats['error'] += 1
            
            if self.logger:
                self.logger.error(f"Failed to predict request from {parts[0]}, request ID {parts[1]}: {str(e)}")
            return False
            
        try:
            self.sock.send_multipart(results)
            
            t_end = time.time()
            t_resp = t_end - t_start
            self.request_stats['last_success'] = t_end
            self.request_stats['response_times'].append(t_resp)
            
        except zmq.error.Again as e:
            self.request_stats['send_failed'] += 1
            
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
        
        dataset = MultiChannelDataset(metadata,
                                      image_cube[:,0,:,:],
                                      image_cube[:,-1,:,:])
        
        results = self.predictor.predict_dataset(dataset)
        
        return client_id, request_id, json.dumps(results).encode()
