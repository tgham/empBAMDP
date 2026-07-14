from logging.config import dictConfig


def configure_logging(logfile):
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        }
    }

    if logfile:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "filename": logfile,
            "mode": "a",
            "formatter": "default",
            "level": "INFO",
        }

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # keep uvicorn/third-party loggers
            "formatters": {
                "default": {
                    "format": "{asctime} - {levelname} - {message}",
                    "style": "{",
                    "datefmt": "%Y-%m-%d %H:%M",
                },
            },
            "handlers": handlers,
            "root": {
                "level": "INFO",
                "handlers": list(handlers.keys()),
            },
            "loggers": {
                # uvicorn loggers
                "uvicorn": {
                    "level": "INFO",
                    "handlers": list(handlers.keys()),
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": "INFO",
                    "handlers": list(handlers.keys()),
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": list(handlers.keys()),
                    "propagate": False,
                },
            },
        }
    )
