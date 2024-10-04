import logging
from datetime import datetime
import os
from pathlib import Path

time = datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
base_path = f"{Path.home()}/.murl"
stamped_path = os.path.join(base_path, time)

try:
    os.makedirs(stamped_path)
    os.remove(f"{base_path}/latest.log")
except OSError as exp:
    print("FATAL ERROR: Could not create log")
    raise exc

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{base_path}/latest.log"),
        logging.FileHandler(f"{stamped_path}/master.log"),
        logging.StreamHandler(),
    ],
)

logger.info("Logger initialized")
