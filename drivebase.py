"""
DriveBaseFramework for Pybricks MicroPython
Supports SPIKE Prime utilizing EV3 codebase
Built by itonasd
"""

from pybricks.parameters import Color
from pybricks.tools import StopWatch
from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor
from micropython import const
from typing import Callable, TypeAlias
from math import pi

condition_t: TypeAlias = Callable[[], bool]
task_t: TypeAlias = Callable[[], None]
pidparams_t: TypeAlias = dict[int, tuple[float, float, float]]

LEFT = const(1)
RIGHT = const(2)

def clamp(val, min_val: float = -1.0, max_val: float = 1.0): return max(min(val, max_val), min_val)
def nearest_hash(s: int, params: pidparams_t) -> int: return min(params.keys(), key=lambda t: abs(t - s))

class PIDController:
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

class MissionMotor:
    def __init__(self, motor: Motor): self.motor = motor
    def move(self, speed: int) -> task_t: return lambda: self.motor.dc(speed)
    def stop(self) -> task_t: return lambda: self.motor.stop()
    def brake(self) -> task_t: return lambda: self.motor.brake()
    def hold(self) -> task_t: return lambda: self.motor.hold()
    def degree(self, target: int) -> condition_t: return lambda: abs(self.motor.angle()) >= target
    def reset_encoder(self) -> None: self.motor.reset_angle(0)

class DriveBaseFramework:
    def __init__(
        self, left_motor: Motor, right_motor: Motor, color_sensor: ColorSensor, hub: PrimeHub,
        forward_params: pidparams_t, linetrace_params: pidparams_t, turn_params: pidparams_t, wheel_diameter: int
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
        self.concurrent_queue = []

    def reset_encoder(self) -> None:
        self.controller.reset()
        self.left_motor.reset_angle(0)
        self.right_motor.reset_angle(0)

    def reset_imu(self) -> None:
        self.target_heading = 0
        self.hub.imu.reset_heading(0)

    def run_async(self, condition: condition_t, task: task_t | None = None, destroying: task_t | None = None) -> None:
        self.concurrent_queue.append([condition, task, destroying])

    def run(self, condition: condition_t, task: task_t, destroying: task_t | None = None) -> None:
        while not condition():
            task()
            for i in range(len(self.concurrent_queue) - 1, -1, -1):
                if not self.concurrent_queue[i][0]():
                    if callable(self.concurrent_queue[i][1]): self.concurrent_queue[i][1]()
                else:
                    if callable(self.concurrent_queue[i][2]): self.concurrent_queue[i][2]()
                    self.concurrent_queue.pop(i)

        if callable(destroying): destroying()
        else: self.stop()()

    def brake(self) -> task_t:
        def callback() -> None:
            self.left_motor.brake()
            self.right_motor.brake()
        return callback
    
    def hold(self) -> task_t:
        def callback() -> None:
            self.left_motor.hold()
            self.right_motor.hold()
        return callback
    
    def stop(self) -> task_t:
        def callback() -> None:
            self.left_motor.stop()
            self.right_motor.stop()
        return callback

    def move_raw(self, left_speed: int, right_speed: int) -> task_t:
        left_speed = clamp(left_speed, -100.0, 100.0)
        right_speed = clamp(right_speed, -100.0, 100.0)
        def callback() -> None:
            self.left_motor.dc(float(left_speed))
            self.right_motor.dc(float(right_speed))
        return callback

    def move_imu(self, speed: int) -> task_t:
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

    def linetrace(self, speed: int, reflection: int) -> task_t:
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

    def turn(self, pivot: int = 0) -> task_t:
        power = [0 if pivot == LEFT else 1, 0 if pivot == RIGHT else 1]
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
    
    def degree(self, target: int) -> condition_t:
        return lambda: (abs(self.left_motor.angle()) + abs(self.right_motor.angle())) / 2 >= target

    def mm(self, target: int) -> condition_t:
        return self.degree(self.mm2deg * target)

    def angle(self, target: int, tolerance: float = 1.0, stable: int = 10) -> condition_t:
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
    
    def black_reflection(self, threshold: int) -> condition_t:
        return lambda: self.color_sensor.reflection() <= threshold
    
    def white_reflection(self, threshold: int) -> condition_t:
        return lambda: self.color_sensor.reflection() >= threshold
    
    def color_reflection(self, color: Color) -> condition_t:
        return lambda: self.color_sensor.color() == color
    
    @staticmethod
    def timer(target: int) -> condition_t:
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
    def forever() -> condition_t:
        return lambda: False

    @staticmethod
    def any(*conditions: condition_t) -> condition_t:
        return lambda: any(c() for c in conditions)

    @staticmethod
    def all(*conditions: condition_t) -> condition_t:
        return lambda: all(c() for c in conditions)
