import logging
import argparse

from oarfish.server import PredictionServer
from oarfish.predict import DualModelPredictor
from oarfish.classify import BinaryLWATVClassifier, MultiLWATVClassifier


logger = logging.getLogger(__name__)


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
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    predictor = DualModelPredictor('models/binary.pt', 'models/multi.pt',
                                   device=device)
    server = PredictionServer(address=args.address, port=args.port,
                              predictor=predictor, logger=logger)
    server.start()
    
    while True:
        try:
            server.receive()
        except KeyboardInterrupt:
            break
            
    server.end()

