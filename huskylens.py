"""
HuskyLens Library for Pybricks MicroPython
Supports EV3 and SPIKE Prime (requires Pybricks 4.0 beta firmware)
Only supports HuskyLens V1, as we does not have V2 hardware
Built by itonasd
"""
 
from pybricks.iodevices import UARTDevice
from pybricks.parameters import Port
from pybricks.tools import wait
from micropython import const
import struct

HEADER = b"\x55\xaa\x11"
CMD_REQUEST_BLOCKS_BY_ID = const(0x27)
CMD_REQUEST_LEARNED = const(0x23)
CMD_RETURN_INFO = const(0x29)
CMD_RETURN_BLOCK = const(0x2A)
CMD_REQUEST_KNOCK = const(0x2C)
CMD_REQUEST_ALGORITHM = const(0x2D)
CMD_RETURN_OK = const(0x2E)
CMD_REQUEST_FIRMWARE_VERSION = const(0x3C)

ALGORITHM_FACE_RECOGNITION = const(0)
ALGORITHM_OBJECT_TRACKING = const(1)
ALGORITHM_OBJECT_RECOGNITION = const(2)
ALGORITHM_LINE_TRACKING = const(3)
ALGORITHM_COLOR_RECOGNITION = const(4)
ALGORITHM_TAG_RECOGNITION = const(5)
ALGORITHM_OBJECT_CLASSIFICATION = const(6)


class Block:
    __slots__ = (
        "x", "y", "width", "height",
        "id", "confidence", "name", "content",
    )

    def __init__(
            self, x: int, y: int, width: int, height: int, 
            id: int, confidence: int = 0, name: str = "", content: str = ""
        ):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.id, self.confidence, self.name, self.content = id, confidence, name, content

    def __repr__(self):
        return (
            "(x=%4d, y=%4d, w=%4d, h=%4d, id=%d, area=%5d, ratio=%.2f)"
            % (self.x, self.y, self.width, self.height, self.id, self.area(), self.ratio())
        )
    
    def cx(self) -> int: return self.x + self.width // 2
    def cy(self) -> int: return self.y + self.height // 2
    def area(self) -> int: return self.width * self.height
    def ratio(self) -> int: return self.width / self.height if self.height else 0


class Huskylens:
    def __init__(self, port: Port):
        self.huskylens = UARTDevice(port, 9600)
        self.cmd_buffer = bytearray(256)

    def checksum(self, data) -> int:
        return sum(data) & 0xFF
    
    def flush(self) -> None:
        for _ in range(10):
            waiting = self.huskylens.waiting()
            if waiting != 0: self.huskylens.read(min(waiting, 64))
            else: break

    def _readv1(self, size: int) -> bytes:
        data = bytearray()
        for _ in range(150):
            if self.huskylens.waiting():
                chunk = self.huskylens.read()
                if chunk: data.extend(chunk)
            if len(data) >= size: return bytes(data[:size])
            wait(1)
        return bytes(data)

    def readv1(self) -> tuple[int | None, bytes | None]:
        window = bytearray()
        for _ in range(100):
            b = self._readv1(1)
            window.extend(b)
            if len(window) > 3: window.pop(0)
            if window == HEADER: break
        else: return None, None

        length = self._readv1(1)[0]
        command = self._readv1(1)[0]
        payload = self._readv1(length) if length else b""
        checksum = self._readv1(1)[0]

        expected = bytearray(HEADER)
        expected.append(length)
        expected.append(command)
        expected.extend(payload)

        if checksum == self.checksum(expected): return command, payload
        else: return None, None

    def cmdv1(self, command: int, payload: bytes | None = None) -> None:
        payload_length = len(payload) if payload else 0
        checksum_pos = 5 + payload_length 
        buffer = self.cmd_buffer

        buffer[0] = 0x55
        buffer[1] = 0xAA
        buffer[2] = 0x11
        buffer[3] = payload_length
        buffer[4] = command

        if payload:
            for i in range(payload_length): buffer[5 + i] = payload[i]
        buffer[checksum_pos] = self.checksum(memoryview(buffer[:checksum_pos]))
        self.huskylens.write(bytes(buffer[: checksum_pos + 1]))

    def ok(self, retries: int = 10) -> bool:
        for _ in range(retries):
            cmd, _ = self.readv1()
            if cmd == CMD_RETURN_OK: return True
            wait(10)
        return False

    def connected(self) -> bool:
        self.flush()
        self.cmdv1(CMD_REQUEST_KNOCK)
        return self.ok()
    
    def algorithm(self, id: int) -> bool:
        payload = bytearray(struct.pack("h", id))
        self.cmdv1(CMD_REQUEST_ALGORITHM, payload)
        return self.ok()
    
    def version(self) -> str | None:
        self.cmdv1(CMD_REQUEST_FIRMWARE_VERSION)
        _, payload = self.readv1()
        if payload: return payload.decode("utf-8")
        return None
    
    def retrieve(self, id: int | None = None) -> list[Block]:
        blocks: list[Block] = []
        if id is not None:
            payload = bytearray(struct.pack("h", id))
            self.cmdv1(CMD_REQUEST_BLOCKS_BY_ID, payload)
        else: self.cmdv1(CMD_REQUEST_LEARNED)

        cmd, info = self.readv1()
        if cmd == CMD_RETURN_INFO:
            n = struct.unpack("h", info[:2])[0] if len(info) >= 2 else 0

            for _ in range(n):
                cmd, data = self.readv1()
                if cmd == CMD_RETURN_BLOCK:
                    obj = Block(*struct.unpack("hhhhh", data))
                    if (id is None or obj.id == id): blocks.append(obj)

        self.flush()
        return blocks
