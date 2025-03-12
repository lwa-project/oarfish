import sys
import time
import logging
import argparse

from oarfish.models import *
from oarfish.server import PredictionServer
from oarfish.predict import DualModelPredictor
from oarfish.classify import BinaryLWATVClassifier, MultiLWATVClassifier


DEFAULT_BINARY = get_default_binary_model()
DEFAULT_MULTI = get_default_multi_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Start up an oarfish server to classify LWATV images over the network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--address', type=str, default='127.0.0.1',
                        help='IP address to bind to')
    parser.add_argument('--port', type=int, default=5555,
                        help='TCP port to listen on')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU to bind to for prediction')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Only use the CPU')
    parser.add_argument('--binary-model', type=str, default=DEFAULT_BINARY,
                        help='binary model to use for prediction')
    parser.add_argument('--multi-model', type=str, default=DEFAULT_MULTI,
                        help='multi-class model to use for prediction')
    parseradd_argument('-r', '--report-interval', type=int, default=600,
                       help='connection statistics report interval (0 disables)')
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
        
    device = f"cuda:{args.gpu}"
    if args.cpu_only:
        device = 'cpu'
    predictor = DualModelPredictor(args.binary_model, args.multi_model,
                                   device=device, logger=logger)
    server = PredictionServer(address=args.address, port=args.port,
                              predictor=predictor, logger=logger)
    server.start()
    
    t_last_report = time.time()
    while True:
        try:
            server.receive()
            
            t_now = time.time()
            if args.report_interval > 0 and (t_now - t_last_report) > args.report_interval:
                t_last_report = t_now
                logger.info("Connection statistics: %s", str(server.get_stats()))
                
        except KeyboardInterrupt:
            break
            
    server.end()
