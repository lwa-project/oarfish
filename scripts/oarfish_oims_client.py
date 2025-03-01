import sys
import time
import numpy as np
import logging
import argparse

from lsl_toolkits.OrvilleImage import OrvilleImageDB

from oarfish.client import PredictionClient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="oarfish client for classification of Orville .oims files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filename', type=str,
                        help='.oims file to classify from')
    parser.add_argument('--address', type=str, default='127.0.0.1',
                        help='IP address to bind to')
    parser.add_argument('--port', type=int, default=5555,
                        help='TCP port to listen on')
    parser.add_argument('--timeout', type=float, default=2.0,
                        help='message timeout in seconds')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='print debug messages as well as info and higher')
    args = parser.parse_args()
    
    logger = logging.StreamHandler(sys.stdout)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        
    
    client = PredictionClient(args.address, args.port,
                              timeout=args.timeout, logger=logger)
    client.start()
    self.logger("Server info: %s", str(client.identify()))
    
    with OrvilleImageDB(sys.argv[1]) as db:
        station = dh.header['station']
        try:
            station = station.decode()
        except AttributeError:
            pass
            
        extra_info = {'station': station}
        if station == 'lwa1':
            extra_info['lon'] = '-107.62835d'
            extra_info['lat'] = '34.068894d'
            extra_info['height'] = '2133.6m'
        elif station == 'lwasv':
            extra_info['lon'] = '-106.885783d'
            extra_info['lat'] = '34.348358d'
            extra_info['height'] = '1477.8m'
        elif station == 'lwana':
            extra_info['lon'] = '-107.640d'
            extra_info['lat'] = '34.247d'
            extra_info['height'] = '2134m'
        else:
            self.log.warn(f"Unknown station '{station}', positions will be suspect")
            
        t0 = time.time()
        for i in range(db.nint):
            info, data = db.read_image(return_masked=False)
            info.update(extra_info)
            data[:,-1,:,:] = np.abs(data[:,-1,:,:])
            data = data[:,[0,-1],:,:]
            
            result = client.send(info, data)
            print(f"Integration {i}: {str(result)}")
        
            if i != 0 and i % 10 == 0:
                t1 = time.time()
                self.log.info("Average speed is %:.1f ints/s (%.3f s per request)", i/(t1-t0), (t1-t0)/i)
                
    client.end()
