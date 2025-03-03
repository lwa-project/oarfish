import zmq
import json
import time
import uuid
import numpy as np
from logging import Logger

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
            
    def _send_and_recieve(self, parts: Optional[List]=None) -> Optional[List]:
        """
        Backend function to send off a request and wait for a reply.
        """
        
        if self.sock is None:
            raise RuntimeError("Need to call start() before sending")
            
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
                            return results
                            
                        else:
                            rrequest_id = rrequest_id.decode()
                            if self.logger:
                                self.logger.info(f"Discarding response to request ID {rrequest_id}")
                                
                except zmq.error.Again:
                    continue
                    
            if self.logger:
                self.logger.warn(f"Request {request_id} timed out after {self.timeout}s")
            return None
            
        except Exception as e:
            print(f"Error on {request_id}: {str(e)}")
            return None
            
        finally:
            try:
                if poller is not None:
                    poller.unregister(self.sock)
            except:
                pass
                
    def identify(self) -> Optional[Dict]:
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
    def _trim_results(results: List[Dict[str,Any]],
                      to_keep: List[str]=['quality_score', 'final_label']):
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
        
    def send(self, metadata: Dict[str,Any], image_cube: np.ndarray,
                   full_output: bool=False) -> Optional[Union[Dict, List[Dict]]]:
        
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
