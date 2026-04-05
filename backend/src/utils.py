import hashlib
import logging
import logging.config
import secrets


class ColoredFormatter(logging.Formatter):
    """Custom formatter với màu sắc và emoji cho logs dễ nhìn hơn"""

    # Màu sắc ANSI
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }

    # Emoji cho từng level
    EMOJI = {
        "DEBUG": "🔍",
        "INFO": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def format(self, record):
        # Lấy màu và emoji cho level
        color = self.COLORS.get(record.levelname, "")
        emoji = self.EMOJI.get(record.levelname, "")

        # Format timestamp ngắn gọn hơn
        log_time = self.formatTime(record, "%H:%M:%S")

        # Format module name ngắn gọn
        module = record.name.split(".")[-1] if "." in record.name else record.name

        # Tạo log message với format đẹp
        log_msg = (
            f"{self.DIM}{log_time}{self.RESET} "
            f"{emoji} {color}{self.BOLD}{record.levelname:8}{self.RESET} "
            f"{self.DIM}[{module}]{self.RESET} "
            f"{record.getMessage()}"
        )

        # Thêm exception info nếu có
        if record.exc_info:
            log_msg += "\n" + self.formatException(record.exc_info)

        return log_msg


def setup_logging():
    """Thiết lập logging với format đẹp và dễ đọc"""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "colored": {
                    "()": ColoredFormatter,
                },
                "simple": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "level": "INFO",
                    "formatter": "colored",
                    "class": "logging.StreamHandler",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console"],
                    "level": "INFO",
                },
            },
        }
    )


def generate_random_string(length=16):
    """
    Generates a random string of the specified length.
    """
    return secrets.token_hex(length // 2)  # Convert to bytes


def generate_request_id(max_length=32):
    """
    Generates a random string and hashes it using SHA-256.
    """
    random_string = generate_random_string()
    h = hashlib.sha256()
    h.update(random_string.encode("utf-8"))
    return h.hexdigest()[: max_length + 1]