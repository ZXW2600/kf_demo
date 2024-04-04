from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
import time
from typing import Union
from typing import Literal
from scipy.integrate import odeint
from dataclasses import dataclass
import sympy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as mplstyle
mplstyle.use('fast')


@dataclass
class GPS:
    x: float
    y: float
    covar_x: float
    covar_y: float
    name: str = "GPS"
    t: float = 0

    def stateVector(self):
        return np.array([self.x, self.y])

    def covarMatrix(self):
        return np.array([[self.covar_x, 0], [0, self.covar_y]])


@dataclass
class Lidar:
    x: float
    y: float
    covar_x: float
    covar_y: float
    name: str = "Lidar"
    t: float = 0

    def stateVector(self):
        return np.array([self.x, self.y])

    def covarMatrix(self):
        return np.array([[self.covar_x, 0], [0, self.covar_y]])


@dataclass
class Radar:
    rho: float
    theta: float
    v: float

    covar_rho: float
    covar_theta: float
    covar_v: float
    name: str = "Radar"
    t: float = 0

    def stateVector(self):
        return np.array([self.rho, self.theta, self.v])

    def covarMatrix(self):
        return np.array([[self.covar_rho, 0, 0], [0, self.covar_theta, 0], [0, 0, self.covar_v]])


@dataclass
class State:
    x: float
    y: float
    v: float
    yaw: float
    yaw_rate: float = 0

    covar_x: float = 0
    covar_y: float = 0
    covar_v: float = 0
    covar_yaw: float = 0
    covar_yaw_rate: float = 0
    t = 0

    slice_x = slice(0, 1)
    slice_y = slice(1, 2)
    slice_v = slice(2, 3)
    slice_yaw = slice(3, 4)

    def state_vector(self):
        return np.array([self.x, self.y, self.v, self.yaw, self.yaw_rate])

    def from_state_vector(self, state_vector):
        self.x = state_vector[0]
        self.y = state_vector[1]
        self.v = state_vector[2]
        self.yaw = state_vector[3]
        self.yaw_rate = state_vector[4]


@dataclass
class Control:
    v_dot: float
    yaw_rate: float
    t: float = 0


SensorDataType = Union[GPS, Lidar, Radar]


class Visualizer:
    def __init__(self, name="Visualizer"):
        # self.key = None
        self.figure = plt.figure(num=name)
        self._createFigure()
        # self.figure.canvas.mpl_connect('key_press_event', self.key_handler)

    def _createFigure(self):
        self.ax = self.figure.add_subplot(111)
        # set figure name

        # set x and y axis have same unit
        self.ax.set_aspect('equal')

    @staticmethod
    def show(timeout=0.1):
        start = time.time()
        plt.show(block=False)
        show = time.time()
        plt.pause(max(0.001, timeout-(show-start)))
        pause = time.time()


class StateVisualizer(Visualizer):
    # This class is responsible for visualizing the state of the system
    # draw x y and covar using ellipse
    # draw v and it covar using arrow, arrow length is v, angle is yaw, and covar is the width of the arrow
    # draw yaw covar using an arc

    def __init__(self):
        Visualizer.__init__(self, "StateVisualizer")
        self.v_ratio = 0.1
        self.v_covar_ratio = 0.1

    def initDrawing(self):
        self.fig, self.ax = plt.subplots()
        # set x and y axis have same unit
        self.ax.set_aspect('equal')

    def draw_state(self, state: State, color="g"):
        self.draw_xy(state, color)
        self.draw_v_arrow(state, color)
        # self.draw_yaw(state, color)

    def draw_gps(self, gps: GPS, color="r"):
        self.ax.plot(gps.x, gps.y, color+"o")
        self.ax.plot(gps.x+gps.covar_x*np.cos(np.linspace(0, 2*np.pi, 100)),
                     gps.y+gps.covar_y*np.sin(np.linspace(0, 2*np.pi, 100)),
                     color, alpha=0.3)

    def draw_lidar(self, lidar: Lidar, color="b"):
        self.ax.plot(lidar.x, lidar.y, color+"o")
        self.ax.plot(lidar.x+lidar.covar_x*np.cos(np.linspace(0, 2*np.pi, 100)),
                     lidar.y+lidar.covar_y *
                     np.sin(np.linspace(0, 2*np.pi, 100)),
                     color, alpha=0.3)

    def draw_xy(self, state: State, color):
        self.ax.plot(state.x, state.y, color+"o")
        self.ax.plot(state.x+state.covar_x*np.cos(np.linspace(0, 2*np.pi, 100)),
                     state.y+state.covar_y *
                     np.sin(np.linspace(0, 2*np.pi, 100)),
                     color, alpha=0.3)

    def draw_v_arrow(self, state: State, color):

        self.ax.arrow(state.x, state.y,
                      self.v_ratio * state.v * np.cos(state.yaw),
                      self.v_ratio * state.v * np.sin(state.yaw),
                      width=self.v_ratio*(state.covar_v+0.1),
                      head_width=self.v_ratio*(state.covar_v+0.2), head_length=self.v_ratio*state.v*0.1,
                      fc="none", ec=color, alpha=0.3,
                      length_includes_head=True
                      )

    def draw_yaw(self, state: State, color):
        self.ax.plot(state.x+state.v*np.cos(state.yaw+state.covar_yaw*np.linspace(0.5, -0.5, 100)),
                     state.y+state.v *
                     np.sin(state.yaw+state.covar_yaw *
                            np.linspace(0.5, -0.5, 100)),
                     color, alpha=0.3)
        self.ax.plot([state.x, state.x+state.v*np.cos(state.yaw+0.5*state.covar_yaw)],
                     [state.y, state.y+state.v*np.sin(state.yaw+0.5*state.covar_yaw)], color, alpha=0.3)
        self.ax.plot([state.x, state.x+state.v*np.cos(state.yaw-0.5*state.covar_yaw)],
                     [state.y, state.y+state.v*np.sin(state.yaw-0.5*state.covar_yaw)], color, alpha=0.3)

    def draw_radar(self, data: Radar, color="b"):
        x = data.rho*np.cos(data.theta)
        y = data.rho*np.sin(data.theta)
        self.ax.plot([0, x], [0, y], color+"-", alpha=0.3)

        # self.ax.plot(data.rho*np.cos(data.theta)+data.covar_rho*np.cos(np.linspace(0, 2*np.pi, 100)),
        #              data.rho*np.sin(data.theta)+data.covar_rho *
        #              np.sin(np.linspace(0, 2*np.pi, 100)),
        #              color, alpha=0.3)


class TimeSerielVisualizer(Visualizer):
    def __init__(self):
        Visualizer.__init__(self, "TimeSerielVisualizer")
        self.sensor_y_dict = {}

    def _createFigure(self):
        self.ax = self.figure.add_subplot(111)

    def draw(self, data: SensorDataType, color_dict={GPS: "r", Lidar: "g", Radar: "b"}, marker_dict={GPS: "o", Lidar: "x", Radar: "s"}):
        if data.name in self.sensor_y_dict.keys():
            y = self.sensor_y_dict[data.name]
        else:
            y = len(self.sensor_y_dict)
            self.sensor_y_dict[data.name] = y
            self.ax.text(0, y, data.name)

        color = color_dict[type(data)]
        marker = marker_dict[type(data)]

        self.ax.scatter(data.t, y, c=color, marker=marker)


class ControlVisualizer(Visualizer):
    def __init__(self, ):
        Visualizer.__init__(self, "ControlVisualizer")

    def _createFigure(self):
        self.ax_v = self.figure.add_subplot(211)
        self.ax_yaw = self.figure.add_subplot(212)
        # set figure name

        # set x and y axis have same unit
        # self.ax_v.set_aspect('equal')
        # self.ax_yaw.set_aspect('equal')

    def draw(self, control: Control):
        self.ax_v.scatter(control.t, control.v_dot, c="r", marker="o")
        self.ax_yaw.scatter(control.t, control.yaw_rate, c="r", marker="o")


class GPSVisualizer(Visualizer):
    def __init__(self):
        Visualizer.__init__(self, "GPSVisualizer")

    def draw(self, data: GPS, color="r"):
        self.ax.plot(data.x, data.y, color+"o")
        self.ax.plot(data.x+data.covar_x*np.cos(np.linspace(0, 2*np.pi, 100)),
                     data.y+data.covar_y *
                     np.sin(np.linspace(0, 2*np.pi, 100)),
                     color, alpha=0.3)


class LidarVisualizer(Visualizer):
    def __init__(self):
        Visualizer.__init__(self, "LidarVisualizer")

    def draw(self, data: Lidar, color="b"):
        self.ax.plot(data.x, data.y, color+"o")
        self.ax.plot(data.x+data.covar_x*np.cos(np.linspace(0, 2*np.pi, 100)),
                     data.y+data.covar_y *
                     np.sin(np.linspace(0, 2*np.pi, 100)),
                     color, alpha=0.3)


class Simulation:
    def __init__(self, init_state: State):
        self.state = init_state

    def step(self, ctrl: Control, dt):
        self.state = self._update_state(self.state, ctrl, dt)
        return self.state

    def getState(self):
        return self.state

    @staticmethod
    def _dot_state(state_vector: np.ndarray, time, ctrl: Control):
        x, y, v, yaw,yaw_rate = state_vector
        v_dot = ctrl.v_dot
        yaw_rate = ctrl.yaw_rate
        return np.array([v*np.cos(yaw), v*np.sin(yaw), v_dot, yaw_rate,0])

    def _update_state(self, state: State, ctrl: Control, dt):
        # run ode
        state_vector = state.state_vector()
        state_vector = odeint(Simulation._dot_state, state_vector, [
                              0, dt], args=(ctrl,))
        state.from_state_vector(state_vector[-1])
        return state


class Sensor:
    SampleTimeCompareResult = Literal["Later", "Now", "Timeout"]
    sensor_name = {GPS: "GPS", Lidar: "Lidar", Radar: "Radar"}
    sensor_cnt = {GPS: 0, Lidar: 0, Radar: 0}

    def __init__(self, sensor_data_type, sensor_hz=100, sensor_time_offset=0, name=None):
        self.sensor_data_type = sensor_data_type
        self.sensor_hz = sensor_hz
        self.sensor_time_offset = sensor_time_offset

        self.sensor_perid = 1/sensor_hz
        self.name = Sensor.sensor_name[sensor_data_type] + \
            str(Sensor.sensor_cnt[sensor_data_type])
        Sensor.sensor_cnt[sensor_data_type] += 1

        self.reset()

    def _addNoise(self, data) -> SensorDataType:
        raise NotImplementedError()

    def _dataFromState(self, state: State) -> SensorDataType:
        raise NotImplementedError()

    def getSensorData(self, state: State) -> SensorDataType:
        data = self._addNoise(self._dataFromState(state))
        data.name = self.name
        data.t = self.t
        return data

    def ifSample(self, t) -> SampleTimeCompareResult:
        if t < self.next_t:
            return "Timeout"
        elif np.abs(t-self.next_t) < 1e-3:
            return "Now"
        else:
            return "Later"

    def time(self):
        return self.t

    def nextSampleTime(self):
        return self.next_t

    def step(self):
        self.t += self.sensor_perid
        self.next_t += self.sensor_perid

    def reset(self):
        self.t = 0
        self.next_t = self.sensor_time_offset


class GPSSensor(Sensor):
    def __init__(self, covar_x, covar_y, sensor_hz=100, sensor_time_offset=0):
        Sensor.__init__(self, GPS, sensor_hz, sensor_time_offset)
        self.cover_x = covar_x
        self.cover_y = covar_y

    def _addNoise(self, data: GPS):
        data.x += np.random.normal(0, self.cover_x)
        data.y += np.random.normal(0, self.cover_y)
        return data

    def _dataFromState(self, state: State):
        data = GPS(state.x, state.y, self.cover_x, self.cover_y)
        return self._addNoise(data)


class LidarSensor(Sensor):
    def __init__(self, covar_x, covar_y, sensor_hz=100, sensor_time_offset=0):
        Sensor.__init__(self, Lidar, sensor_hz, sensor_time_offset)
        self.cover_x = covar_x
        self.cover_y = covar_y

    def _addNoise(self, data: Lidar) -> Lidar:
        data.x += np.random.normal(0, self.cover_x)
        data.y += np.random.normal(0, self.cover_y)
        return data

    def _dataFromState(self, state: State):
        data = Lidar(state.x, state.y, self.cover_x, self.cover_y)
        return self._addNoise(data)


class RadarSensor(Sensor):
    def __init__(self, covar_rho, covar_theta, covar_v, sensor_hz=100, sensor_time_offset=0):
        Sensor.__init__(self, Radar, sensor_hz, sensor_time_offset)
        self.cover_rho = covar_rho
        self.cover_theta = covar_theta
        self.cover_v = covar_v

    def _addNoise(self, data: Radar):
        data.rho += np.random.normal(0, self.cover_rho)
        data.theta += np.random.normal(0, self.cover_theta)
        data.v += np.random.normal(0, self.cover_v)
        return data

    def _dataFromState(self, state: State):
        rho = np.sqrt(state.x**2+state.y**2)
        theta = np.arctan2(state.y, state.x)
        # map state v to radar v
        rho_vector = np.array([state.x, state.y])/rho
        speed_vector = np.array(
            [state.v*np.cos(state.yaw), state.v*np.sin(state.yaw)])
        v = np.dot(rho_vector, speed_vector)
        data = Radar(rho=rho, theta=theta, v=v, covar_rho=self.cover_rho,
                     covar_theta=self.cover_theta, covar_v=self.cover_v)
        return self._addNoise(data)


class DataSynthesizer:
    def __init__(self, init_state: State, sensors: list[Sensor]):

        self.sensor_hz = [sensor.sensor_hz for sensor in sensors]
        self.sensor_period = [1/sensor.sensor_hz for sensor in sensors]
        self.sensor_time_offset = [
            sensor.sensor_time_offset for sensor in sensors]

        self.init_state = init_state
        self.sim = Simulation(init_state)
        self.sensors = sensors

    def reset(self):
        self.sim = Simulation(self.init_state)

    def get_data(self):
        return self.data

    def _flushSensor(self, t, state: State):
        sensors: list[tuple[float, State, SensorDataType]] = []
        for sensor in self.sensors:
            if sensor.ifSample(t) == "Now":
                sensor_data: GPS | Lidar | Radar = sensor.getSensorData(state)
                sensor_data.t = t
                sensors.append((t, state, sensor_data))
                sensor.step()
        return sensors

    def _getNextSampleTime(self):
        next_sample_time = [sensor.nextSampleTime() for sensor in self.sensors]
        return min(next_sample_time)

    def controlPolicy(self, t: float, pre_state: State):
        return Control(
            v_dot=0,
            yaw_rate=2*(np.sin(t*10)-0.5),
            t=t
        )

    def dataBatch(self, sample_num):

        t = 0
        state = self.sim.getState()
        sensors_data = self._flushSensor(t, state)
        for sensor_data in sensors_data:
            yield *sensor_data, Control(0, 0, 0)

        for i in range(sample_num):
            t_nxt = self._getNextSampleTime()
            dt = t_nxt - t
            control = self.controlPolicy(t_nxt, state)
            state = self.sim.step(control, dt)
            sensors_data = self._flushSensor(t_nxt, state)
            for sensor_data in sensors_data:
                yield *sensor_data, control
            t = t_nxt


class UKF:
    def __init__(self, init_state: State) -> None:
        x = init_state.state_vector()
        dim_x = len(x)
        P = np.diag([init_state.covar_x, init_state.covar_y,
                     init_state.covar_v, init_state.covar_yaw, init_state.covar_yaw_rate])
        points = MerweScaledSigmaPoints(dim_x, alpha=0.1, beta=2, kappa=1)
        self.ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=2, dt=0.1, hx=None,
                                         fx=self.fx,
                                         points=points)
        self.ukf.x = x
        self.ukf.P = P
        self.ukf.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01])

    def predict(self, dt):
        self.ukf.predict(dt, fx=self.fx)

    def update(self, z: SensorDataType):
        if isinstance(z, GPS):
            print("update gps")
            self.update_gps(z)
        elif isinstance(z, Lidar):
            print("update Lidar")
            self.update_lidar(z)
        else:
            print("update Radar")
            self.update_radar(z)

    def update_gps(self, gps: GPS):
        z = gps.stateVector()
        R = gps.covarMatrix()
        self.ukf.update(z=z, R=R, hx=self.h_gps)

    def update_lidar(self, lidar: Lidar):
        z = lidar.stateVector()
        R = lidar.covarMatrix()
        self.ukf.update(z=z, R=R, hx=self.h_lidar)

    def update_radar(self, radar: Radar):

        z = radar.stateVector()
        R = radar.covarMatrix()
        self.ukf.update(z=z, R=R, hx=self.h_radar)

    def fx(self, x, dt):
        p_x, p_y, v, yaw,yaw_rate = x
        p_x_dot = v*np.cos(yaw)
        p_y_dot = v*np.sin(yaw)
        v_dot = 0
        x_dot = np.array([p_x_dot, p_y_dot, v_dot, yaw_rate,0])
        return x+x_dot*dt

    def h_gps(self, x: np.ndarray):
        return np.array([x[State.slice_x].item(), x[State.slice_y].item()])

    def h_lidar(self, x: np.ndarray):
        return np.array([x[State.slice_x].item(), x[State.slice_y].item()])

    def h_radar(self, x):

        rho = np.sqrt(x[State.slice_x].item()**2+x[State.slice_y].item()**2)
        theta = np.arctan2(x[State.slice_y].item(), x[State.slice_x].item())

        # map state v to radar v
        rho_vector = np.array(
            [x[State.slice_x].item(), x[State.slice_y].item()])/rho

        speed_vector = np.array(
            [x[State.slice_v].item() * np.cos(x[State.slice_yaw].item()), x[State.slice_v].item()*np.sin(x[State.slice_yaw].item())])
        v = np.dot(rho_vector, speed_vector)
        return np.array([rho, theta, v])

    def getState(self):
        state = State(
            x=self.ukf.x[State.slice_x].item(),
            y=self.ukf.x[State.slice_y].item(),
            v=self.ukf.x[State.slice_v].item(),
            yaw=self.ukf.x[State.slice_yaw].item(),
            covar_x=self.ukf.P[State.slice_x, State.slice_x].item(),
            covar_y=self.ukf.P[State.slice_y, State.slice_y].item(),
            covar_v=self.ukf.P[State.slice_v, State.slice_v].item(),
            covar_yaw=self.ukf.P[State.slice_yaw, State.slice_yaw].item()

        )

        return state


class EKF:
    def __init__(self) -> None:
        pass

    def predict():
        pass

    def update():
        pass

    def getState():
        pass


class TestCase:
    def __init__(self, filter: Union[UKF, EKF], vis_dt=0.1, sim_time_factor=5, vis_ctrl=False, vis_time=False, vis_state=True):
        self.filter = filter
        self.sensors = [
            GPSSensor(0.1, 0.1, sensor_hz=10, sensor_time_offset=0.01),
            LidarSensor(0.05, 0.03, sensor_hz=20, sensor_time_offset=0),
            RadarSensor(0.01, 0.03, 0.01, sensor_hz=50,
                        sensor_time_offset=0.02)
        ]
        self.datasynthesizer = DataSynthesizer(
            State(
                x=0, y=0, v=1, yaw=0
            ),
            self.sensors
        )
        if vis_time:
            self.time_vis = TimeSerielVisualizer()
        if vis_state:
            self.state_vis = StateVisualizer()
        if vis_ctrl:
            self.ctrl_vis = ControlVisualizer()

        self.vis_time = vis_time
        self.vis_state = vis_state
        self.vis_ctrl = vis_ctrl

        self.vis_dt = vis_dt
        self.sim_time_factor = sim_time_factor

    def run(self, sample_num=10000):
        last_t = 0
        last_vis_t = 0
        for (t, state, sensor_data, ctrl) in self.datasynthesizer.dataBatch(sample_num):
            dt = t-last_t
            last_t = t
            dt_vis = t-last_vis_t

            self.filter.predict(dt)
            predic_state = self.filter.getState()

            self.filter.update(sensor_data)
            update_state = self.filter.getState()

            if self.vis_time:
                self.time_vis.draw(sensor_data)
            if self.vis_ctrl:
                self.ctrl_vis.draw(ctrl)

            if self.vis_state:

                if isinstance(sensor_data, GPS):
                    self.state_vis.draw_gps(sensor_data, "r")
                elif isinstance(sensor_data, Lidar):
                    self.state_vis.draw_lidar(sensor_data, "g")
                else:
                    self.state_vis.draw_radar(sensor_data, "m")

                if True or dt_vis > self.vis_dt:
                    # self.state_vis.draw_state(last_state, "c")
                    self.state_vis.draw_state(state, 'k')
                    # print(f"predict {predic_state}")
                    # self.state_vis.draw_state(predic_state, "b")
                    self.state_vis.draw_state(update_state, "c")
                    last_vis_t = t
            if self.vis_state or self.vis_ctrl or self.vis_time:
                Visualizer.show(0.05)


if __name__ == "__main__":
    ukf = UKF(State(x=0, y=0, v=1, yaw=0, yaw_rate=0,
                    covar_x=0.1, covar_y=0.1, covar_v=0.1, covar_yaw=0.1, covar_yaw_rate=0.1))

    test = TestCase(filter=ukf, vis_dt=0.1, sim_time_factor=2)
    test.run(100000)
