"""by lyuwenyu
"""

import os 
import sys 
import logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("train.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def main(args) -> None:
    '''main
    '''
    try:
        logger.info("Initializing distributed mode...")
        dist.init_distributed()
        
        if args.seed is not None:
            logger.info(f"Setting seed: {args.seed}")
            dist.set_seed(args.seed)

        assert not all([args.tuning, args.resume]), \
            'Only support from_scratch or resume or tuning at one time'

        logger.info(f"Loading configuration from {args.config}")
        cfg = YAMLConfig(
            args.config,
            resume=args.resume, 
            use_amp=args.amp,
            tuning=args.tuning
        )

        logger.info("Configuration loaded successfully")
        logger.debug(f"Configuration details: {cfg.yaml_cfg}")

        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        
        if args.test_only:
            logger.info("Running in test-only mode")
            solver.val()
        else:
            logger.info("Starting training")
            solver.fit()
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config file')
    parser.add_argument('--resume', '-r', type=str, help='Path to the checkpoint file to resume from')
    parser.add_argument('--tuning', '-t', type=str, help='Path to the tuning file')
    parser.add_argument('--test-only', action='store_true', default=False, help='Run in test-only mode')
    parser.add_argument('--amp', action='store_true', default=False, help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, help='Seed for reproducibility')
    args = parser.parse_args()

    logger.info("Starting script with arguments: %s", args)
    main(args)