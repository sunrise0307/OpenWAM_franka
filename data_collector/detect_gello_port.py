"""Select the connected GELLO port configured for the teleoperation agent."""

import sys
from pathlib import Path

from teleop.agents.teleop_agent import PORT_CONFIG_MAP


def select_port(configured_ports, exists=Path.exists):
    connected = [port for port in configured_ports if exists(Path(port))]
    if len(connected) != 1:
        raise ValueError(
            f"Expected one connected GELLO port in PORT_CONFIG_MAP, found {len(connected)}: "
            + ", ".join(connected)
        )
    return connected[0]


if __name__ == "__main__":
    try:
        print(select_port(PORT_CONFIG_MAP))
    except ValueError as exc:
        print(f"GELLO auto-detection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
