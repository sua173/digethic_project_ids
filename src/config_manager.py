import os
import configparser
from datetime import datetime, timezone
import pytz

# Constants
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
DEFAULT_IDLE_TIMEOUT = 10
DEFAULT_ACTIVE_TIMEOUT = 120
DEFAULT_TIMEZONE = "America/Halifax"


class ConfigurationManager:
    """Configuration management class"""

    @staticmethod
    def get_config():
        """Get the current configuration values."""
        config = configparser.ConfigParser()

        if os.path.exists(CONFIG_PATH):
            config.read(CONFIG_PATH)
            print("Config file loaded")
            idle_timeout = int(
                config.get(
                    "nfstream", "idle_timeout", fallback=str(DEFAULT_IDLE_TIMEOUT)
                )
            )
            active_timeout = int(
                config.get(
                    "nfstream", "active_timeout", fallback=str(DEFAULT_ACTIVE_TIMEOUT)
                )
            )
        else:
            idle_timeout = DEFAULT_IDLE_TIMEOUT
            active_timeout = DEFAULT_ACTIVE_TIMEOUT
            ConfigurationManager._create_config_file(
                config, idle_timeout, active_timeout
            )

        return idle_timeout, active_timeout

    @staticmethod
    def _create_config_file(config, idle_timeout, active_timeout):
        """Create a new configuration file"""
        print("Config file not found. Creating new config file...")
        config["nfstream"] = {
            "idle_timeout": str(idle_timeout),
            "active_timeout": str(active_timeout),
        }
        with open(CONFIG_PATH, "w") as configfile:
            config.write(configfile)


class TimeUtils:
    """Time conversion utilities"""

    @staticmethod
    def ms_timestamp_to_localtime(ms_timestamp, tz_str=DEFAULT_TIMEZONE):
        """Convert millisecond timestamp to local time"""
        sec = ms_timestamp / 1000
        dt_utc = datetime.fromtimestamp(sec, tz=timezone.utc)
        tz = pytz.timezone(tz_str)
        return dt_utc.astimezone(tz)

    @staticmethod
    def localtime_to_ms_timestamp(dt_local, tz_str=DEFAULT_TIMEZONE):
        """Convert local time to millisecond timestamp"""
        if dt_local.tzinfo is None:
            tz = pytz.timezone(tz_str)
            dt_local = tz.localize(dt_local)
        dt_utc = dt_local.astimezone(pytz.utc)
        return int(dt_utc.timestamp() * 1000)


def get_config():
    return ConfigurationManager.get_config()


def ms_timestamp_to_localtime(ms_timestamp, tz_str=DEFAULT_TIMEZONE):
    return TimeUtils.ms_timestamp_to_localtime(ms_timestamp, tz_str)


def localtime_to_ms_timestamp(dt_local, tz_str=DEFAULT_TIMEZONE):
    return TimeUtils.localtime_to_ms_timestamp(dt_local, tz_str)
