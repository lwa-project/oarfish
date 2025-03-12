import zmq
import json
import time
import uuid
import numpy as np
from logging import Logger
from collections import deque

from typing import Optional, Any, Dict, List, Union

class PredictionClient:
    """
    ZeroMQ-based client to pass Sokes I and |V| images or image cubes to a
    prediction server for classification.
    """
    
    def __init__(self, address: str='127.0.0.1', port: int=5555,
                       timeout: float=1.0, logger: Optional[Logger]=None):
        self.address = address
        self.port = port
        self.timeout = timeout
        self.logger = logger
        
        self.client_id = str(uuid.uuid4())
        self.ctx = None
        self.sock = None
        
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
        """
        Connect to the prediction server.
        """
        
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.DEALER)
        
        self.sock.setsockopt(zmq.IDENTITY, self.client_id.encode())
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, int(self.timeout*1000))
        self.sock.setsockopt(zmq.RECONNECT_IVL, 100)
        self.sock.setsockopt(zmq.RECONNECT_IVL_MAX, 10000)
        
        self._reset_stats()
        self.request_stats['start_time'] = time.time()
        
        self.sock.connect(f"tcp://{self.address}:{self.port}")
        
        if self.logger:
            self.logger.info(f"Client ID is {self.client_id}")
            self.logger.info(f"Connected to prediction server at {self.address} port {self.port}")
            
    def end(self):
        """
        Disconnect from the prediction server.
        """
        
        self.sock.close()
        self.ctx.term()
        
        self.ctx = None
        self.sock = None
        
        if self.logger:
            self.logger.info("Disconnected from prediction server")
            
    def get_stats() -> Dict[str, Any]:
        """
        Get connection statistics about the client that include:
         * how long it has been since start() was called
         * how many requests have been processed
         * when the last requests was successfully processed
         * the average processing time of the last 50 requests
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
            
        return {'client_id': self.client_id,
                'connected': self.ctx is not None,
                'uptime': uptime,
                'requests': {'total': self.request_stats['total'],
                             'send_failed': self.request_stats['send_failed']
                             'error': self.request_stats['error']
                             'timeout': self.request_stats['timeout']
                             'successful': self.request_stats['success']
                            }
                'last_successful_response': last_good,
                'average_response_time': resp_time
               }
        
    def _send_and_recieve(self, parts: Optional[List[bytes]]=None) -> Optional[bytes]:
        """
        Backend function to send off a request and wait for a reply.
        """
        
        if self.sock is None:
            raise RuntimeError("Need to call start() before sending")
            
        t_start = time.time()
        self.request_stats['total'] += 1
        
        request_id = str(uuid.uuid4()).encode()
        if parts is None:
            parts = []
        parts = [request_id,] + parts
        
        poller = None
        try:
            poller = zmq.Poller()
            poller.register(self.sock, zmq.POLLIN)
            
            try:
                self.sock.send_multipart(parts)
            except zmq.error.Again as e:
                self.request_stats['send_failed'] += 1
                
                if self.logger:
                    self.logger.warn(f"Failed to send to prediction server: {str(e)}")
                return None
                
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    events = dict(poller.poll(100))
                    
                    if self.sock in events and events[self.sock] == zmq.POLLIN:
                        rrequest_id, results = self.sock.recv_multipart()
                        if rrequest_id == request_id:
                            t_end = time.time()
                            t_resp = t_end - t_start
                            self.request_stats['last_success'] = t_end
                            self.request_stats['response_times'].append(t_resp)
                            self.request_stats['success'] += 1
                            
                            return results
                            
                        else:
                            rrequest_id = rrequest_id.decode()
                            if self.logger:
                                self.logger.info(f"Discarding response to request ID {rrequest_id}")
                                
                except zmq.error.Again:
                    continue
                    
            self.request_stats['timeout'] += 1
            
            if self.logger:
                self.logger.warn(f"Request {request_id} timed out after {self.timeout}s")
            return None
            
        except Exception as e:
            self.request_stats['error'] += 1
            
            print(f"Error on {request_id}: {str(e)}")
            return None
            
        finally:
            try:
                if poller is not None:
                    poller.unregister(self.sock)
            except:
                pass
                
    def identify(self) -> Optional[Dict[str, Any]]:
        """
        Query the prediction server for information about how it makes
        predictions.
        
        If there is a problem querying the server None is returned instead.
        """
        
        ident = self._send_and_recieve()
        if ident:
            ident = json.loads(ident)
        return ident
        
    @staticmethod
    def _trim_results(results: List[Dict[str, Any]],
                      to_keep: List[str]=['quality_score', 'final_label']) -> List[Dict[str, Any]]:
        """
        Prune a full results dictionary down to only the requested keys.
        """
        
        trimmed_results = []
        for result in results:
            trimmed = {}
            for key in to_keep:
                try:
                    trimmed[key] = result[key]
                except KeyError:
                    pass
            trimmed_results.append(trimmed)
        return trimmed_results
        
    def send(self, metadata: Dict[str, Any], image_cube: np.ndarray,
                   full_output: bool=False) -> Optional[Union[Dict[str, Any],
                                                              List[Dict[str, Any]]]]:
        
        """
        Given a metadata dictionary and an array containing Stokes I and |V|
        data, send the data to the prediction server to get an overall quality
        score and label for the data.  If the data is 3D [Stokes by X by Y] a
        single dictionary is returned.  Multi-channel data [channel by Stokes
        by X by Y] will return a list of dictionaries, one for each channel.
        
        If there is a problem querying the server None is returned instead.
        
        For the full prediction server output set `full_output` to True.
        """
        
        reshapeNeeded = False
        if len(image_cube.shape) == 2:
            reshapeNeeded = True
            image_cube = image_cube.reshape(1, *image_cube.shape)
        metadata['image_cube_shape'] = image_cube.shape
        
        metadata = json.dumps(metadata).encode()
        image_cube = image_cube.tobytes()
        
        results = self._send_and_recieve([metadata, image_cube])
        if results:
            results = json.loads(results)
            if not full_output:
                results = self._trim_results(results)
            if reshapeNeeded:
                results = results[0]
        return results
