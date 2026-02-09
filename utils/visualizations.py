"""Real-time visualization windows for safety filter monitoring.

This module provides PyQt5-based visualization windows for monitoring the
safety filter during deployment or simulation. The windows update in
real-time and display:

- Robot state and LiDAR data (StateLidarWindow)
- Safety filter metrics: value, controls, disturbance bounds (SafetyFilterWindow)
- MPS planner: occupancy map and planned trajectory (MPSWindow)

Requirements:
    pip install pyqt5 pyqtgraph

Example usage:
    from PyQt5.QtWidgets import QApplication
    from utils.visualizations import SafetyFilterWindow

    # Create shared history dict that gets updated by control loop
    history = {
        'times': [], 'values': [], 'v_xs': [], 'v_yaws': [], 'dsts': [],
        'states': [], 'raw_angles': [], 'raw_lidars': [], 'raw_qualities': [],
        'counts': [], 'inp_lidars': [], 'lidar_states': [],
    }

    # Create Qt application and window
    app = QApplication([])
    window = SafetyFilterWindow(
        title="Safety Filter Monitor",
        update_period=0.1,
        control_min=[0, -2],
        control_max=[2, 2],
        value_threshold=0.35,
        time_span=10.0,
        history=history,
    )
    window.show()

    # Run control loop in separate thread, updating history dict
    # The window will automatically refresh based on update_period
"""

import numpy as np

try:
    import pyqtgraph as pg
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QTransform
    from PyQt5.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


def check_pyqt_available():
    """Check if PyQt5 and pyqtgraph are available."""
    if not PYQT_AVAILABLE:
        raise ImportError(
            "PyQt5 and pyqtgraph are required for visualization. "
            "Install with: pip install pyqt5 pyqtgraph"
        )


class StateLidarWindow(QMainWindow):
    """Real-time visualization of robot state and LiDAR data.

    This window displays the robot's position, heading, and LiDAR scan
    in either world frame (raw) or ego frame (input to network).
    """

    def __init__(
        self,
        title: str,
        update_period: float,
        vis_type: str,
        plot_radius: float,
        rel_lidar_position: list,
        quality_threshold: float,
        max_range: float,
        time_span: float,
        history: dict,
    ):
        """Initialize the state and LiDAR visualization window.

        Args:
            title: Window title.
            update_period: Update interval in seconds.
            vis_type: Visualization type - "raw" (world frame) or "input" (ego frame).
            plot_radius: Plot radius in meters.
            rel_lidar_position: LiDAR position relative to robot [x, y].
            quality_threshold: Minimum quality for valid LiDAR points.
            time_span: Time span of trajectory to display in seconds.
            history: Shared dict containing logged data with keys:
                - times: List of timestamps
                - states: List of [x, y, theta] states
                - raw_angles: List of raw LiDAR angle arrays
                - raw_lidars: List of raw LiDAR range arrays
                - raw_qualities: List of raw LiDAR quality arrays
                - counts: List of valid point counts
                - inp_lidars: List of preprocessed LiDAR arrays
                - lidar_states: List of LiDAR reference states
        """
        check_pyqt_available()
        super().__init__()

        self.setWindowTitle(title)
        self.vis_type = vis_type
        assert vis_type in ["raw", "input"], "vis_type must be 'raw' or 'input'"

        self.plot_radius = plot_radius
        self.rel_lidar_position = rel_lidar_position
        self.quality_threshold = quality_threshold
        self.max_range = max_range
        self.time_span = time_span
        self.history = history
        self.origin = [0, 0]

        # Create plot items
        self.position_item = pg.PlotDataItem()
        self.heading_item = pg.PlotDataItem()
        self.lidar_item = pg.ScatterPlotItem()

        # Style plot items
        self.position_item.setBrush((0, 255, 0))
        self.position_item.setSymbol("o")
        self.position_item.setSymbolSize(8)
        self.heading_item.setBrush((0, 0, 255))
        self.lidar_item.setBrush((255, 0, 0))

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addItem(self.position_item)
        self.plot_widget.addItem(self.heading_item)
        self.plot_widget.addItem(self.lidar_item)
        self.plot_widget.getPlotItem().setLabel("bottom", text="x (m)")
        self.plot_widget.getPlotItem().setLabel("left", text="y (m)")
        self.plot_widget.getPlotItem().getViewBox().setAspectLocked(True)

        if vis_type == "input":
            r = self.max_range * 1.1
            self.plot_widget.setXRange(-r, r)
            self.plot_widget.setYRange(-r, r)

        # Create state label
        self.state_label = QLabel()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.state_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(int(update_period * 1000))

    def _update_plot(self):
        """Update the plot with latest data."""
        if len(self.history["times"]) == 0:
            return

        times = np.asarray(self.history["times"])
        states = np.asarray(self.history["states"])
        states = states[times > (times[-1] - self.time_span)]

        if self.vis_type == "raw":
            self._update_raw_view(states)
        else:
            self._update_input_view(states)

    def _update_raw_view(self, states: np.ndarray):
        """Update visualization in world frame."""
        x, y, th = states[-1]

        # Plot trajectory
        self.position_item.setData(states[:, :2])
        self.heading_item.setData([x, x + np.cos(th)], [y, y + np.sin(th)])

        # Plot LiDAR in world frame (from LIDAR_SCAN, may not be available yet)
        if len(self.history["raw_angles"]) == 0:
            self.state_label.setText(f"x: {x:.2f}, y: {y:.2f}, theta: {th:.2f}")
            return

        abs_lidar_x = (
            x
            + self.rel_lidar_position[0] * np.cos(th)
            - self.rel_lidar_position[1] * np.sin(th)
        )
        abs_lidar_y = (
            y
            + self.rel_lidar_position[0] * np.sin(th)
            + self.rel_lidar_position[1] * np.cos(th)
        )

        raw_angle = np.asarray(self.history["raw_angles"][-1])
        raw_lidar = np.asarray(self.history["raw_lidars"][-1])
        raw_qualities = np.asarray(self.history["raw_qualities"][-1])

        abs_angle = raw_angle + th
        lidar_xs = raw_lidar * np.cos(abs_angle) + abs_lidar_x
        lidar_ys = raw_lidar * np.sin(abs_angle) + abs_lidar_y

        # Filter by quality
        is_valid = np.logical_and(
            raw_qualities > self.quality_threshold,
            np.arange(len(raw_qualities)) < self.history["counts"][-1],
        )
        self.lidar_item.setData(lidar_xs[is_valid], lidar_ys[is_valid])

        # Auto-pan plot to follow robot
        ox, oy = self.origin
        tol_radius = self.plot_radius * 3 / 4
        if (x - ox) > tol_radius:
            ox = x - tol_radius
        if (x - ox) < -tol_radius:
            ox = x + tol_radius
        if (y - oy) > tol_radius:
            oy = y - tol_radius
        if (y - oy) < -tol_radius:
            oy = y + tol_radius
        self.origin = [ox, oy]

        self.plot_widget.setXRange(ox - self.plot_radius, ox + self.plot_radius)
        self.plot_widget.setYRange(oy - self.plot_radius, oy + self.plot_radius)

        self.state_label.setText(f"x: {x:.2f}, y: {y:.2f}, theta: {th:.2f}")

    def _update_input_view(self, states: np.ndarray):
        """Update visualization in ego frame."""
        if len(self.history["lidar_states"]) == 0:
            return

        lidar_state = np.asarray(self.history["lidar_states"][-1])
        cth, sth = np.cos(lidar_state[2]), np.sin(lidar_state[2])
        rot = np.array([[cth, sth], [-sth, cth]])

        # Transform states to ego frame
        rel_states = states.copy()
        rel_states[:, :2] = np.matmul(
            rot, (rel_states[:, :2] - lidar_state[:2])[:, :, np.newaxis]
        ).squeeze(axis=-1)
        rel_states[:, 2] = rel_states[:, 2] - lidar_state[2]
        rel_states[:, 2] = (rel_states[:, 2] + np.pi) % (2 * np.pi) - np.pi

        x, y, th = rel_states[-1]

        # Plot trajectory in ego frame
        self.position_item.setData(rel_states[:, :2])
        self.heading_item.setData([x, x + np.cos(th)], [y, y + np.sin(th)])

        # Plot preprocessed LiDAR (from SAFETY_STATUS, may not be available yet)
        if len(self.history["inp_lidars"]) > 0:
            inp_lidar = np.asarray(self.history["inp_lidars"][-1])
            thetas = np.linspace(-np.pi, np.pi, len(inp_lidar), endpoint=False)
            self.lidar_item.setData(inp_lidar * np.cos(thetas), inp_lidar * np.sin(thetas))

        self.state_label.setText(f"x: {x:.2f}, y: {y:.2f}, theta: {th:.2f}")


class SafetyFilterWindow(QMainWindow):
    """Real-time visualization of safety filter metrics.

    Displays time-series plots of:
    - Predicted value function
    - Control commands (v_x, v_yaw)
    - Estimated disturbance bounds
    """

    def __init__(
        self,
        title: str,
        update_period: float,
        control_min: list,
        control_max: list,
        value_threshold: float,
        time_span: float,
        history: dict,
    ):
        """Initialize the safety filter visualization window.

        Args:
            title: Window title.
            update_period: Update interval in seconds.
            control_min: Minimum control bounds [v_x_min, v_yaw_min].
            control_max: Maximum control bounds [v_x_max, v_yaw_max].
            value_threshold: Value threshold for safety filtering.
            time_span: Time span to display in seconds.
            history: Shared dict containing logged data with keys:
                - times: List of timestamps
                - values: List of predicted values
                - v_xs: List of forward velocity commands
                - v_yaws: List of yaw rate commands
                - dsts: List of [dst_dxdy_max, dst_dth_max] disturbance bounds
        """
        check_pyqt_available()
        super().__init__()

        self.setWindowTitle(title)
        self.time_span = time_span
        self.history = history

        # Value plot
        self.value_plot_widget = pg.PlotWidget()
        self.value_plot_widget.getPlotItem().setLabel(
            "bottom", text="Value (m) vs Time (s)"
        )
        self.value_plot_widget.getPlotItem().addLine(
            y=value_threshold, pen=pg.mkPen((255, 255, 255), width=2)
        )
        self.value_plot_widget.setXRange(-time_span, 0)
        self.value_plot = self.value_plot_widget.plot(pen=pg.mkPen((0, 255, 0), width=2))

        # Forward velocity plot
        self.v_x_plot_widget = pg.PlotWidget()
        self.v_x_plot_widget.getPlotItem().setLabel(
            "bottom", text="Forward Velocity (m/s) vs Time (s)"
        )
        self.v_x_plot_widget.getPlotItem().addLine(
            y=0, pen=pg.mkPen((255, 255, 255), width=1)
        )
        self.v_x_plot_widget.setXRange(-time_span, 0)
        self.v_x_plot_widget.setYRange(control_min[0], control_max[0])
        self.v_x_plot = self.v_x_plot_widget.plot(pen=pg.mkPen((0, 200, 255), width=2))

        # Yaw rate plot
        self.v_yaw_plot_widget = pg.PlotWidget()
        self.v_yaw_plot_widget.getPlotItem().setLabel(
            "bottom", text="Yaw Rate (rad/s) vs Time (s)"
        )
        self.v_yaw_plot_widget.getPlotItem().addLine(
            y=0, pen=pg.mkPen((255, 255, 255), width=1)
        )
        self.v_yaw_plot_widget.setXRange(-time_span, 0)
        self.v_yaw_plot_widget.setYRange(control_min[1], control_max[1])
        self.v_yaw_plot = self.v_yaw_plot_widget.plot(pen=pg.mkPen((255, 200, 0), width=2))

        # XY disturbance bound plot
        self.dst_xy_plot_widget = pg.PlotWidget()
        self.dst_xy_plot_widget.getPlotItem().setLabel(
            "bottom", text="Disturbance Bound XY (m/s) vs Time (s)"
        )
        self.dst_xy_plot_widget.setXRange(-time_span, 0)
        self.dst_xy_plot_widget.setYRange(0, 1)
        self.dst_xy_plot = self.dst_xy_plot_widget.plot(pen=pg.mkPen((255, 100, 100), width=2))

        # Theta disturbance bound plot
        self.dst_th_plot_widget = pg.PlotWidget()
        self.dst_th_plot_widget.getPlotItem().setLabel(
            "bottom", text="Disturbance Bound Theta (rad/s) vs Time (s)"
        )
        self.dst_th_plot_widget.setXRange(-time_span, 0)
        self.dst_th_plot_widget.setYRange(0, 2)
        self.dst_th_plot = self.dst_th_plot_widget.plot(pen=pg.mkPen((255, 100, 255), width=2))

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.value_plot_widget)
        layout.addWidget(self.v_x_plot_widget)
        layout.addWidget(self.v_yaw_plot_widget)
        layout.addWidget(self.dst_xy_plot_widget)
        layout.addWidget(self.dst_th_plot_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(int(update_period * 1000))

    def _update_plot(self):
        """Update all plots with latest data."""
        if len(self.history["filter_times"]) == 0:
            return

        # Use filter_times (from SAFETY_STATUS) not times (from ROBOT_STATE)
        times = np.asarray(self.history["filter_times"])
        values = np.asarray(self.history["values"])
        v_xs = np.asarray(self.history["v_xs"])
        v_yaws = np.asarray(self.history["v_yaws"])
        dsts = np.asarray(self.history["dsts"])

        # Relative time (0 = now)
        times = times - times[-1]

        # Only show last time_span seconds
        in_time_span = times > -self.time_span
        times = times[in_time_span]
        values = values[in_time_span]
        v_xs = v_xs[in_time_span]
        v_yaws = v_yaws[in_time_span]
        dsts = dsts[in_time_span]

        self.value_plot.setData(times, values)
        self.v_x_plot.setData(times, v_xs)
        self.v_yaw_plot.setData(times, v_yaws)
        self.dst_xy_plot.setData(times, dsts[:, 0])
        self.dst_th_plot.setData(times, dsts[:, 1])


class MPSWindow(QMainWindow):
    """Real-time visualization of MPS planner.

    Displays the SLAM occupancy map and the planned trajectory.
    """

    DEFAULT_MAP_SIZE_PIXELS = 200
    DEFAULT_MAP_SIZE_METERS = 20

    def __init__(
        self,
        title: str,
        update_period: float,
        mps_info: dict,
        history: dict,
        map_size_pixels: int = None,
        map_size_meters: float = None,
    ):
        """Initialize the MPS visualization window.

        Args:
            title: Window title.
            update_period: Update interval in seconds.
            mps_info: Shared dict containing planner data with keys:
                - occupancy_map: 2D numpy array of occupancy grid
                - optimal_state_sequence: Array of planned states [[x, y, th], ...]
            map_size_pixels: Map resolution in pixels (default: 200).
            map_size_meters: Map size in meters (default: 20).
        """
        check_pyqt_available()
        super().__init__()

        self.setWindowTitle(title)
        self.mps_info = mps_info
        self.history = history

        map_size_pixels = map_size_pixels or self.DEFAULT_MAP_SIZE_PIXELS
        map_size_meters = map_size_meters or self.DEFAULT_MAP_SIZE_METERS

        # Create transform for occupancy grid
        transform = QTransform()
        transform.scale(map_size_meters / map_size_pixels, map_size_meters / map_size_pixels)
        transform.translate(-map_size_pixels / 2, -map_size_pixels / 2)

        # Create occupancy map image
        self.occupancy_item = pg.ImageItem(levels=(0, 1))
        self.occupancy_item.setTransform(transform)

        # Create trajectory plot
        self.trajectory_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=2))

        # Create robot position marker
        self.robot_item = pg.ScatterPlotItem(
            brush=(0, 150, 255), symbol="o", size=10
        )

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addItem(self.occupancy_item)
        self.plot_widget.addItem(self.trajectory_item)
        self.plot_widget.addItem(self.robot_item)
        self.plot_widget.getPlotItem().setLabel("bottom", text="x (m)")
        self.plot_widget.getPlotItem().setLabel("left", text="y (m)")
        self.plot_widget.getPlotItem().getViewBox().setAspectLocked(True)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(int(update_period * 1000))

    def _update_plot(self):
        """Update the plot with latest data."""
        occupancy_map = self.mps_info.get("occupancy_map")
        optimal_sequence = self.mps_info.get("optimal_state_sequence")

        if occupancy_map is None or optimal_sequence is None:
            return

        self.occupancy_item.setImage(np.asarray(occupancy_map))
        self.trajectory_item.setData(np.asarray(optimal_sequence)[:, :2])

        if len(self.history["states"]) > 0:
            state = self.history["states"][-1]
            self.robot_item.setData([state[0]], [state[1]])
