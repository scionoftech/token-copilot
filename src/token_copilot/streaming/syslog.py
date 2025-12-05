"""Syslog streaming with RFC 5424 support."""

import logging
import socket
import json
from typing import Optional
from datetime import datetime

from .base import BaseStreamer, StreamEvent

logger = logging.getLogger(__name__)


class SyslogStreamer(BaseStreamer):
    """
    Stream cost events to syslog servers (RFC 5424).

    Supports both TCP and UDP protocols with structured data.

    Example:
        ```python
        from token_copilot import TokenPilotCallback
        from token_copilot.streaming import SyslogStreamer

        # TCP syslog
        streamer = SyslogStreamer(
            host="logs.company.com",
            port=514,
            protocol="tcp"
        )

        # UDP syslog
        streamer = SyslogStreamer(
            host="127.0.0.1",
            port=514,
            protocol="udp"
        )

        # With custom facility and severity
        streamer = SyslogStreamer(
            host="logs.company.com",
            port=514,
            facility=16,  # Local0
            severity=6,   # Informational
            app_name="token-copilot"
        )

        callback = TokenPilotCallback(streamer=streamer)
        ```
    """

    # Syslog severity levels (RFC 5424)
    EMERGENCY = 0
    ALERT = 1
    CRITICAL = 2
    ERROR = 3
    WARNING = 4
    NOTICE = 5
    INFORMATIONAL = 6
    DEBUG = 7

    # Syslog facilities (RFC 5424)
    KERN = 0
    USER = 1
    MAIL = 2
    DAEMON = 3
    AUTH = 4
    SYSLOG = 5
    LPR = 6
    NEWS = 7
    UUCP = 8
    CRON = 9
    AUTHPRIV = 10
    FTP = 11
    LOCAL0 = 16
    LOCAL1 = 17
    LOCAL2 = 18
    LOCAL3 = 19
    LOCAL4 = 20
    LOCAL5 = 21
    LOCAL6 = 22
    LOCAL7 = 23

    def __init__(
        self,
        host: str = "localhost",
        port: int = 514,
        protocol: str = "udp",
        facility: int = 16,  # LOCAL0
        severity: int = 6,   # INFORMATIONAL
        app_name: str = "token-copilot",
        hostname: Optional[str] = None,
        format: str = "rfc5424",  # or "json"
        enabled: bool = True,
    ):
        """
        Initialize syslog streamer.

        Args:
            host: Syslog server hostname (default: "localhost")
            port: Syslog server port (default: 514)
            protocol: "tcp" or "udp" (default: "udp")
            facility: Syslog facility (0-23, default: 16=LOCAL0)
            severity: Syslog severity (0-7, default: 6=INFORMATIONAL)
            app_name: Application name in syslog message (default: "token-copilot")
            hostname: Hostname for syslog message (default: auto-detect)
            format: Message format "rfc5424" or "json" (default: "rfc5424")
            enabled: Whether streaming is enabled (default: True)
        """
        super().__init__(enabled=enabled)
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.facility = facility
        self.severity = severity
        self.app_name = app_name
        self.hostname = hostname or socket.gethostname()
        self.format = format

        # Create socket
        self._socket: Optional[socket.socket] = None
        self._connect()

    def _connect(self):
        """Connect to syslog server."""
        try:
            if self.protocol == "tcp":
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.port))
            else:  # udp
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.debug(f"Connected to syslog server {self.host}:{self.port} ({self.protocol})")
        except Exception as e:
            logger.error(f"Failed to connect to syslog server: {e}")
            self._socket = None

    def _format_rfc5424(self, event: StreamEvent) -> str:
        """
        Format event as RFC 5424 syslog message.

        Format: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID STRUCTURED-DATA MSG
        """
        # Calculate priority
        priority = (self.facility * 8) + self.severity

        # RFC 5424 timestamp format
        timestamp = event.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Structured data
        sd_elements = []
        if event.user_id:
            sd_elements.append(f'user_id="{event.user_id}"')
        if event.org_id:
            sd_elements.append(f'org_id="{event.org_id}"')
        if event.model:
            sd_elements.append(f'model="{event.model}"')
        if event.cost is not None:
            sd_elements.append(f'cost="{event.cost:.6f}"')
        if event.input_tokens is not None:
            sd_elements.append(f'input_tokens="{event.input_tokens}"')
        if event.output_tokens is not None:
            sd_elements.append(f'output_tokens="{event.output_tokens}"')

        structured_data = f'[llm_cost {" ".join(sd_elements)}]' if sd_elements else "-"

        # Message
        msg = f"LLM cost event: {event.model or 'unknown'} - ${event.cost or 0:.6f}"

        # Build RFC 5424 message
        message = (
            f"<{priority}>1 {timestamp} {self.hostname} {self.app_name} - - "
            f"{structured_data} {msg}"
        )

        return message

    def _format_json(self, event: StreamEvent) -> str:
        """Format event as JSON for JSON-over-syslog."""
        priority = (self.facility * 8) + self.severity
        data = event.to_dict()
        data['_priority'] = priority
        data['_hostname'] = self.hostname
        data['_app_name'] = self.app_name
        return json.dumps(data)

    def send(self, event: StreamEvent) -> bool:
        """
        Send event to syslog server.

        Args:
            event: StreamEvent to send

        Returns:
            True if sent successfully, False on error
        """
        if not self._socket:
            self._connect()
            if not self._socket:
                return False

        try:
            # Format message
            if self.format == "json":
                message = self._format_json(event)
            else:
                message = self._format_rfc5424(event)

            # Send message
            message_bytes = message.encode('utf-8')

            if self.protocol == "tcp":
                # TCP requires newline terminator
                message_bytes += b'\n'
                self._socket.sendall(message_bytes)
            else:  # udp
                self._socket.sendto(message_bytes, (self.host, self.port))

            logger.debug(f"Sent syslog message: {len(message_bytes)} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to send syslog message: {e}")
            # Try to reconnect
            self._socket = None
            return False

    def close(self):
        """Close syslog connection."""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
