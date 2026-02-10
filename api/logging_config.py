import logging
from .config import LOG_LEVEL


class RequestIdFilter(logging.Filter):
    def filter(self,record:logging.LogRecord)-> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

logging.basicConfig(
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format = "%(asctime)s | %(levelname)s | %(name)s| %(request_id)s | %(message)s"
)

for h in logging.getLogger().handlers:
    h.addFilter(RequestIdFilter())

base_logger = logging.getLogger("intelligent-reasoner")
