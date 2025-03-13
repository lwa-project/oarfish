import sys
import time
import numpy as np
import logging
import argparse

from lsl_toolkits.OrvilleImage import OrvilleImageDB

from oarfish.client import PredictionClient
from oarfish.utils import station_to_earthlocation


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
    
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        
    
    client = PredictionClient(args.address, args.port,
                              timeout=args.timeout, logger=logger)
    client.start()
    logger.info("Server info: %s", str(client.identify()))
    
    with OrvilleImageDB(sys.argv[1]) as db:
        station = db.header.station
        if isinstance(station, bytes):
            station = station.decode()
        station = station.lower()
            
        extra_info = {'station': station}
        try:
            el = station_to_earthlocation(station)
            extra_info['lon'] = f"{el.lon.to('deg').value}d"
            extra_info['lat'] = f"{el.lat.to('deg').value}d"
            extra_info['height'] = f"{el.height.to('m').value}m"
        except ValueError:
            logger.warn(f"Unknown station '{station}', positions will be suspect")
            
        t0 = time.time()
        for i in range(db.nint):
            info, data = db.read_image()
            info.update(extra_info)
            for key in info:
                if isinstance(info[key], bytes):
                    info[key] = info[key].decode()
            data = data.data
            data[:,-1,:,:] = np.abs(data[:,-1,:,:])
            data = data[:,[0,-1],:,:]
            
            result = client.send(info, data)
            print(f"Integration {i}: {str(result)}")
        
            if i != 0 and i % 10 == 0:
                t1 = time.time()
                logger.info("Average speed is %.1f ints/s (%.3f s per request)", i/(t1-t0), (t1-t0)/i)
                
    print(f"Connection Statistics: {str(client.get_stats())}")
    
    client.end()
