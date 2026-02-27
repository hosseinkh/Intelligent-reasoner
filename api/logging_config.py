import logging
from pythonjsonlogger import jsonlogger

from .config import LOG_LEVEL


# 1) Adapter that MERGES request_id with per-log extra={...}
class MergeExtraAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        merged = dict(self.extra or {})
        if "extra" in kwargs and kwargs["extra"]:
            merged.update(kwargs["extra"])
        kwargs["extra"] = merged
        return msg, kwargs


# 2) Ensure request_id always exists (so formatter won't crash)
class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True
    

# 3) Formatter that keeps ALL extra fields automatically
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # normalize some names (optional)
        log_record.setdefault("message", record.getMessage())
        log_record.setdefault("level", record.levelname)
        log_record.setdefault("logger", record.name)

        # request_id should be present because of the filter, but just in case:
        log_record.setdefault("request_id", getattr(record, "request_id", "-"))


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # important: clear handlers so uvicorn reload doesn't duplicate logs
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())

    # IMPORTANT: use YOUR formatter, not jsonlogger.JsonFormatter()
    handler.setFormatter(CustomJsonFormatter())

    root.addHandler(handler)
    
# Your app can import this as base logger factory
base_logger = logging.getLogger("intelligent-reasoner")    