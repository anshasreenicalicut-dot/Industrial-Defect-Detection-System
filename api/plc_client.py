
import os


def publish_detection(result):
    enabled = (
        os.getenv("PLC_ENABLED", "false").lower()
        == "true"
    )

    if not enabled:
        return {
            "enabled": False,
            "sent": False,
            "message": "PLC integration is disabled."
        }

    host = os.getenv("PLC_HOST", "").strip()

    if not host:
        raise RuntimeError("PLC_HOST is not configured.")

    from pymodbus.client import ModbusTcpClient

    port = int(os.getenv("PLC_PORT", "502"))
    unit_id = int(os.getenv("PLC_UNIT_ID", "1"))
    start_address = int(
        os.getenv("PLC_REGISTER_START", "0")
    )

    detections = result.get("detections", [])
    first = detections[0] if detections else None

    bbox = (
        first["bbox_xyxy"]
        if first
        else {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
    )

    def uint16(value):
        return max(0, min(65535, int(value)))

    registers = [
        uint16(result.get("detection_count", 0)),
        uint16(first["class_id"] if first else 0),
        uint16(
            round(first["confidence"] * 1000)
            if first else 0
        ),
        uint16(bbox["x1"]),
        uint16(bbox["y1"]),
        uint16(bbox["x2"]),
        uint16(bbox["y2"])
    ]

    client = ModbusTcpClient(
        host=host,
        port=port,
        timeout=2
    )

    try:
        if not client.connect():
            raise ConnectionError("PLC connection failed.")

        try:
            response = client.write_registers(
                address=start_address,
                values=registers,
                device_id=unit_id
            )
        except TypeError:
            response = client.write_registers(
                address=start_address,
                values=registers,
                slave=unit_id
            )

        if response.isError():
            raise RuntimeError("PLC returned a Modbus error.")

        return {
            "enabled": True,
            "sent": True,
            "register_start": start_address
        }

    finally:
        client.close()
