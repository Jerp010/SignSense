#!/usr/bin/env python3
"""Quick test of the new logging format."""

import sys
sys.path.insert(0, 'd:\\Jeff Code\\SignSense\\signsense')

from utils.logger import logger, log_session_start, log_session_end, log_init, log_success, log_warning, log_error

# Test the logging
log_session_start()

logger.info("")
logger.info("Testing various log levels and formats:")
logger.info("")

log_init("Camera System", "640x480 resolution")
log_success("Camera initialized successfully | delay=43.35s")

logger.info("Normal info message")
logger.debug("Debug details about operation")
logger.warning("Example warning message")

log_warning("This is a warning")

log_session_end()
