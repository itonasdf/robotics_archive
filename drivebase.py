"""
DriveBaseFramework for Pybricks MicroPython
Supports SPIKE Prime utilizing EV3 codebase
Built by itonasd
"""

from pybricks.parameters import Color
from pybricks.tools import StopWatch
from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor
from pybricks.tools import wait
from micropython import const
from math import pi

PIVOT_LEFT = const(1)
PIVOT_RIGHT = const(2)

def clamp(val, min_val: float = -1.0, max_val: float = 1.0): return max(min(val, max_val), min_val)
def nearest_hash(s: int, params: dict[int]) -> int: return min(params.keys(), key=lambda t: abs(t - s))

class PIDController:
    __slots__ = (
        "kp", "ki", "kd", "integral_limit", "integral_deadzone", "dt", "integral", "prev_error"
    )

    def __init__(self):
        self.kp = 1
        self.ki = 0
        self.kd = 0.1
        self.integral_limit = 10
        self.integral_deadzone = 10
        self.dt = 0.01
        self.integral = 0.0
        self.prev_error = 0.0

    def setPID(self, params: tuple[float, float, float]) -> None:
        self.kp, self.ki, self.kd = params

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def calculate(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement
        if abs(error) <= self.integral_deadzone:
            self.integral += error * self.dt
            self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

class ConcurrentTask:
    __slots__ = (
        "wait", "until", "task", "cleanup", "started"
    )
    
    def __init__(self, wait, until, task, cleanup):
        self.wait = wait
        self.until = until
        self.task = task
        self.cleanup = cleanup
        self.started = False

    def update(self) -> bool:
        if not self.started: self.started = self.wait()
        if not self.started: return False
        if self.until():
            if callable(self.cleanup): self.cleanup()
            return True

        self.task()
        return False

class MissionMotor:
    __slots__ = ("motor")

    def __init__(self, motor: Motor): self.motor = motor
    def move(self, speed: int): return lambda: self.motor.dc(speed)
    def stop(self): return lambda: self.motor.stop()
    def brake(self): return lambda: self.motor.brake()
    def hold(self): return lambda: self.motor.hold()
    def degree(self, target: int): return lambda: abs(self.motor.angle()) >= target
    def resetEncoder(self) -> None: self.motor.reset_angle(0)

class DriveBaseFramework:
    def __init__(
        self, left_motor: Motor, right_motor: Motor, color_sensor: ColorSensor, hub: PrimeHub,
        forward_params, linetrace_params, turn_params, wheel_diameter: int, operate_frequency: int
    ):
        self.target_heading = 0
        self.controller = PIDController()
        self.hub = hub
        self.forward_params = forward_params
        self.linetrace_params = linetrace_params
        self.turn_params = turn_params
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.color_sensor = color_sensor
        self.mm2deg = 360 / (pi * wheel_diameter)
        self.dt = 1 / operate_frequency
        self.controller.dt = self.dt
        self.concurrent_queue: list[ConcurrentTask] = []

    def resetEnconder(self) -> None:
        self.controller.reset()
        self.left_motor.reset_angle(0)
        self.right_motor.reset_angle(0)

    def getEncoder(self) -> int:
        return (abs(self.left_motor.angle()) + abs(self.right_motor.angle())) / 2

    def resetImu(self) -> None:
        self.target_heading = 0
        self.hub.imu.reset_heading(0)

    def runConcurrent(self, wait, until, task, cleanup = None) -> None:
        self.concurrent_queue.append(ConcurrentTask(wait, until, task, cleanup))

    def run(self, until, task, cleanup = None) -> None:
        while not until():
            for i in range(len(self.concurrent_queue) - 1, -1, -1):
                if (self.concurrent_queue[i].update()): self.concurrent_queue.pop(i)

            task()
            wait(self.dt)
        if callable(cleanup): cleanup()
        else: self.stop()()
    
    def brake(self):
        def callback() -> None:
            self.left_motor.brake()
            self.right_motor.brake()
        return callback
    
    def hold(self):
        def callback() -> None:
            self.left_motor.hold()
            self.right_motor.hold()
        return callback
    
    def stop(self):
        def callback() -> None:
            self.left_motor.stop()
            self.right_motor.stop()
        return callback

    def moveRaw(self, left_speed: int, right_speed: int):
        left_speed = clamp(left_speed, -100.0, 100.0)
        right_speed = clamp(right_speed, -100.0, 100.0)
        def callback() -> None:
            self.left_motor.dc(float(left_speed))
            self.right_motor.dc(float(right_speed))
        return callback

    def moveImu(self, speed: int):
        started = False
        speed = clamp(speed, -100.0, 100.0)
        def callback() -> None:
            nonlocal started
            if not started:
                self.controller.setPID(self.forward_params[nearest_hash(speed, self.forward_params)])
                started = True

            rotation = self.controller.calculate(self.target_heading, self.hub.imu.heading())
            left = speed + rotation
            right = speed - rotation
            maximum = max(abs(left), abs(right))
            if maximum > 100:
                left *= 100 / maximum
                right *= 100 / maximum
            
            self.left_motor.dc(float(left))
            self.right_motor.dc(float(right))
        return callback

    def linetrace(self, speed: int, reflection: int):
        started = False
        speed = clamp(speed, -100.0, 100.0)
        def callback() -> None:
            nonlocal started
            if not started:
                self.controller.setPID(self.linetrace_params[nearest_hash(speed, self.linetrace_params)])
                started = True

            rotation = self.controller.calculate(reflection, self.color_sensor.reflection())
            left = speed + rotation
            right = speed - rotation
            maximum = max(abs(left), abs(right))
            if maximum > 100:
                left *= 100 / maximum
                right *= 100 / maximum
            
            self.left_motor.dc(float(left))
            self.right_motor.dc(float(right))
        return callback

    def turn(self, pivot: int = 0):
        power = [0 if pivot == PIVOT_LEFT else 1, 0 if pivot == PIVOT_RIGHT else 1]
        started = False
        def callback() -> None:
            nonlocal started
            if not started:
                self.controller.reset()
                turn_angle = abs(self.target_heading - self.hub.imu.heading())
                self.controller.setPID(self.turn_params[nearest_hash(turn_angle, self.turn_params)])
                started = True

            rotation = self.controller.calculate(self.target_heading, self.hub.imu.heading())
            self.left_motor.dc(float(clamp(rotation, -100, 100) * power[0]))
            self.right_motor.dc(float(clamp(-rotation, -100, 100) * power[1]))
        return callback
    
    def degree(self, target: int):
        return lambda: self.getEncoder() >= target

    def mm(self, target: int):
        return self.degree(self.mm2deg * target)

    def angle(self, target: int, tolerance: float = 1.0, stable: int = 10):
        n = 0
        started = False
        def callback() -> bool:
            nonlocal n, started
            if not started:
                self.target_heading = target
                started = True

            if abs(self.target_heading - self.hub.imu.heading()) <= tolerance: n += 1
            else: n = 0
            return n >= stable
        return callback
    
    def blackReflection(self, threshold: int):
        return lambda: self.color_sensor.reflection() <= threshold
    
    def whiteReflection(self, threshold: int):
        return lambda: self.color_sensor.reflection() >= threshold
    
    def colorReflection(self, color: Color):
        return lambda: self.color_sensor.color() == color
    
    @staticmethod
    def timer(target: int):
        timer = StopWatch()
        started = False
        def callback() -> bool:
            nonlocal started
            if not started:
                timer.reset()
                started = True
            return timer.time() >= target
        return callback
    
    @staticmethod
    def forever():
        return lambda: False

    @staticmethod
    def any(*conditions):
        return lambda: any(c() for c in conditions)

    @staticmethod
    def all(*conditions):
        return lambda: all(c() for c in conditions)
