"""
utils/logger.py
===============
Centralized logging for SignSense.
Tracks initialization times, errors, and performance metrics for debugging.
"""

import logging
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Optional, Any, Callable


# ---------------------------------------------------------------------------
# Setup logging directories
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"signsense_{TIMESTAMP}.log"


# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    COLORS = {
        'DEBUG':    '\033[36m',      # Cyan
        'INFO':     '\033[32m',      # Green
        'WARNING':  '\033[33m',      # Yellow
        'ERROR':    '\033[31m',      # Red
        'CRITICAL': '\033[35m',      # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS and sys.stdout.isatty():
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str = "SignSense") -> logging.Logger:
    """
    Initialize the logger with both file and console handlers.
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler (always log everything)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(levelname)-8s | %(name)-15s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


# ---------------------------------------------------------------------------
# Timing decorators and context managers
# ---------------------------------------------------------------------------

class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        """
        Parameters
        ----------
        name : str
            Description of the operation being timed
        log_level : int
            Logging level to use when reporting time
        """
        self.name = name
        self.log_level = log_level
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.log(self.log_level, f"START: {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        
        if exc_type is not None:
            logger.error(
                f"FAILED: {self.name} | {exc_type.__name__}: {exc_val} | Time: {elapsed:.4f}s"
            )
            return False
        
        # Choose label and level based on duration
        if elapsed > 5.0:
            label = "SLOW DONE"
            level = logging.WARNING
        elif elapsed > 1.0:
            label = "DONE"
            level = logging.INFO
        else:
            label = "DONE"
            level = logging.DEBUG
            
        logger.log(level, f"{label}: {self.name} | Time: {elapsed:.4f}s")
        return True


def time_it(name: Optional[str] = None, log_level: int = logging.INFO):
    """
    Decorator to time function execution.
    
    Parameters
    ----------
    name : str, optional
        Custom name for logging. If None, uses function name.
    log_level : int
        Logging level for timing reports
        
    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            operation_name = name or func.__name__
            with TimingContext(operation_name, log_level):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Track and report on performance metrics."""
    
    def __init__(self):
        self.timings = {}  # {operation_name: [times]}
        self.errors = {}   # {operation_name: count}
        
    def record_timing(self, operation: str, duration: float):
        """Record an operation's duration."""
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)
        
    def record_error(self, operation: str, error_msg: str):
        """Record an error occurrence."""
        if operation not in self.errors:
            self.errors[operation] = 0
        self.errors[operation] += 1
        logger.error(f"Error in {operation}: {error_msg}")
        
    def get_stats(self, operation: str) -> dict:
        """Get statistics for an operation."""
        if operation not in self.timings:
            return {}
            
        times = self.timings[operation]
        return {
            'count': len(times),
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times),
            'total': sum(times),
        }
        
    def report(self):
        """Generate a performance report."""
        logger.info("=" * 70)
        logger.info("PERFORMANCE REPORT")
        logger.info("=" * 70)
        
        # Timing stats
        if self.timings:
            logger.info("\nTiming Statistics:")
            for op in sorted(self.timings.keys()):
                stats = self.get_stats(op)
                logger.info(
                    f"  {op:30s} | Count: {stats['count']:3d} | "
                    f"Avg: {stats['avg']:8.4f}s | "
                    f"Min: {stats['min']:8.4f}s | Max: {stats['max']:8.4f}s"
                )
        
        # Error summary
        if self.errors:
            logger.warning("\nErrors Encountered:")
            for op in sorted(self.errors.keys()):
                logger.warning(f"  {op:30s} | Count: {self.errors[op]}")
        
        logger.info("=" * 70)


# Global performance tracker
perf_tracker = PerformanceTracker()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def log_init(module_name: str, details: str = ""):
    """Log module initialization."""
    msg = f"Initializing {module_name}"
    if details:
        msg += f" - {details}"
    logger.info(f"INIT: {msg}")


def log_error(component: str, error: Exception, context: str = ""):
    """Log an error with context."""
    msg = f"Error in {component}: {type(error).__name__}: {str(error)}"
    if context:
        msg += f" [{context}]"
    logger.error(msg)


def log_success(message: str):
    """Log a successful operation."""
    logger.info(f"SUCCESS: {message}")


def log_warning(message: str):
    """Log a warning."""
    logger.warning(f"WARNING: {message}")


# ---------------------------------------------------------------------------
# Session info
# ---------------------------------------------------------------------------

def log_session_start():
    """Log application startup information."""
    logger.info("=" * 70)
    logger.info("SIGNSENSE SESSION STARTED")
    logger.info("=" * 70)
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info("=" * 70)


def log_session_end():
    """Log application shutdown and generate report."""
    logger.info("=" * 70)
    logger.info("SIGNSENSE SESSION ENDED")
    logger.info("=" * 70)
    perf_tracker.report()
