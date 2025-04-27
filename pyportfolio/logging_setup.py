import logging
import logging.config
import yaml
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'pyportfolio', 'config', 'logging_config.yaml')

def setup_logging(config_path=_DEFAULT_CONFIG_PATH):
    """Loads logging configuration from the specified YAML file."""
    path = config_path
    
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.info("Successfully applied dictConfig.") 
                logging.getLogger().info("Logging configured successfully from YAML.")
            except Exception as e:
                logging.error(f"ERROR loading/applying logging config from {path}: {e}", file=sys.stderr) # Add print
                logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logging.error(f"Error loading logging config from {path}: {e}. Falling back to basicConfig.")
    else:
        logging.warning(f"WARNING: Logging config file not found at {path}. Falling back to basicConfig.", file=sys.stderr) # Add print
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.warning(f"Logging configuration file not found at {path}. Falling back to basicConfig.")

setup_logging()