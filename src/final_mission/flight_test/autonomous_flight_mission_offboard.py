import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

import geometry_utils
import vehicle_command


class OffboardWaypointNode(Node):
    """
    Launch passes parameters via YAML (Node(parameters=[param_file]) ).
    This node:
      1) Reads wp1..wp4 (lat/lon/alt_rel) & takeoff_alt_rel, speed/tolerances from ROS 2 parameters.
      2) Latches EKF origin from VehicleLocalPosition.ref_lat/lon/alt.
      3) Latches wp0 at ARM using VehicleLocalPosition (local NED) only.
      4) Takes off vertically to takeoff_alt_rel (Z tol = 1 m),
      5) Flies WP1→WP4 with constant XY speed using TrajectorySetpoint (position-only),
         yaw pointing to current target (XY tol = 2 m, Z tol = 1 m),
      6) Returns to wp0 and LAND.
    """

    def __init__(self):
        super().__init__('offboard_waypoint')

        # QoS (PX4 typical)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.sp_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # Subscribers
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_pos_cb, qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos)

        # ---- Parameters (declared to read from YAML) ----
        self.declare_parameter('speed_xy', 2.0)          # m/s, constant XY speed
        self.declare_parameter('xy_tolerance', 2.0)      # m, horizontal tolerance
        self.declare_parameter('z_tolerance', 1.0)       # m, vertical tolerance
        self.declare_parameter('takeoff_alt_rel', 10.0)  # m, relative to ARM z (up positive)
        # Waypoints wp1..wp4 as dotted parameters (lat/lon/alt_rel)
        for i in range(1, 5):
            self.declare_parameter(f'wp{i}.lat', 0.0)
            self.declare_parameter(f'wp{i}.lon', 0.0)
            self.declare_parameter(f'wp{i}.alt_rel', 0.0)

        self.speed_xy = float(self.get_parameter('speed_xy').value)
        self.xy_tol = float(self.get_parameter('xy_tolerance').value)
        self.z_tol = float(self.get_parameter('z_tolerance').value)
        self.takeoff_alt_rel = float(self.get_parameter('takeoff_alt_rel').value)

        # ---- Internal state ----
        self.local_pos: VehicleLocalPosition | None = None
        self.status: VehicleStatus | None = None

        # EKF local origin (deg/deg/m) and radians cache
        self.origin_set = False
        self.ref_lat_deg = None
        self.ref_lon_deg = None
        self.ref_alt = None
        self.ref_lat_rad = None
        self.ref_lon_rad = None

        # ARM origin (wp0) from VehicleLocalPosition at ARM time
        self.arm_set = False
        self.wp0_ned = None    # {'x':..,'y':..,'z':..}
        self.wp0_z = None      # z at ARM (for alt_rel)

        # Waypoints from params and converted NED
        self.wp_list_raw = []  # [{'lat':..,'lon':..,'alt_rel':..}, ...]
        self.wps_ned = []      # [{'x':..,'y':..,'z':..}] -> WP1..WP4
        self.takeoff_ned = None

        # State machine
        self.state = 'WAIT_ORIGINS'  # wait EKF origin + params
        self.leg = 0                 # 0..len(wps_ned)-1
        self.state_changed = True

        # Interpolation (position-only constant speed)
        self.start_time_ns = None
        self.duration_ns = None
        self.start_wp = None  # dict x,y,z
        self.goal_wp = None   # dict x,y,z

        # Timer at 20 Hz
        self.timer = self.create_timer(0.05, self.timer_cb)

        # Load params into raw list
        self._load_params()
        self.get_logger().info('Offboard waypoint node initialized.')

    # ---------- Parameter loader ----------
    def _load_params(self):
        self.wp_list_raw.clear()
        for i in range(1, 5):
            lat = float(self.get_parameter(f'wp{i}.lat').value)
            lon = float(self.get_parameter(f'wp{i}.lon').value)
            alt_rel = float(self.get_parameter(f'wp{i}.alt_rel').value)
            # accept only if lat/lon are non-zero (basic sanity)
            if lat != 0.0 or lon != 0.0:
                self.wp_list_raw.append({'lat': lat, 'lon': lon, 'alt_rel': alt_rel})
        if not self.wp_list_raw:
            self.get_logger().warn('No waypoints provided via parameters (wp1..wp4).')

    # ---------- Subscribers ----------
    def local_pos_cb(self, msg: VehicleLocalPosition):
        self.local_pos = msg
        # Latch EKF origin once available
        if not self.origin_set and getattr(msg, 'xy_global', False):
            self.ref_lat_deg = float(msg.ref_lat)  # degrees
            self.ref_lon_deg = float(msg.ref_lon)
            self.ref_alt = float(msg.ref_alt)      # meters AMSL
            self.ref_lat_rad = math.radians(self.ref_lat_deg)
            self.ref_lon_rad = math.radians(self.ref_lon_deg)
            self.origin_set = True
            self.get_logger().info(
                f'EKF origin lat/lon/alt set: {self.ref_lat_deg:.7f}, {self.ref_lon_deg:.7f}, {self.ref_alt:.2f}')

    def status_cb(self, msg: VehicleStatus):
        # Detect ARM rising edge and latch wp0 from local position
        prev_arming = self.status.arming_state if self.status is not None else None
        self.status = msg
        if prev_arming is not None and prev_arming != VehicleStatus.ARMING_STATE_ARMED \
           and msg.arming_state == VehicleStatus.ARMING_STATE_ARMED and not self.arm_set:
            if self.local_pos is None or not self.origin_set:
                return  # will retry in timer
            self.wp0_ned = {
                'x': float(self.local_pos.x),
                'y': float(self.local_pos.y),
                'z': float(self.local_pos.z),
            }
            self.wp0_z = float(self.local_pos.z)
            self.arm_set = True
            self.get_logger().info(
                f'ARM latched: wp0_ned=({self.wp0_ned["x"]:.2f}, {self.wp0_ned["y"]:.2f}, {self.wp0_ned["z"]:.2f})')

    # ---------- Helpers ----------
    def _build_ned_waypoints_if_ready(self):
        if not (self.origin_set and self.arm_set and self.wp_list_raw and self.wp0_ned is not None):
            return False
        # Build WP1..WP4 in NED using alt_rel from ARM altitude (use local z at ARM)
        self.wps_ned = []
        for wp in self.wp_list_raw:
            x, y = geometry_utils.wgs84_to_ned(wp['lat'], wp['lon'],
                                               self.ref_lat_rad, self.ref_lon_rad)
            z = self.wp0_z - float(wp['alt_rel'])  # NED(+down): up is negative
            self.wps_ned.append({'x': x, 'y': y, 'z': z})
        # Takeoff target directly above wp0
        self.takeoff_ned = {
            'x': self.wp0_ned['x'],
            'y': self.wp0_ned['y'],
            'z': self.wp0_z - float(self.takeoff_alt_rel),
        }
        return True

    def _publish_position_sp(self, x, y, z, yaw):
        vehicle_command.publish_heartbeat_ob_pos_sp(self.ctrl_mode_pub, self.get_clock())
        vehicle_command.publish_position_setpoint(self.sp_pub, x, y, z, yaw, self.get_clock())

    def _leg_init(self, start_wp: dict, goal_wp: dict, speed_xy: float):
        self.start_wp = start_wp.copy()
        self.goal_wp = goal_wp.copy()
        dx = goal_wp['x'] - start_wp['x']
        dy = goal_wp['y'] - start_wp['y']
        xy_dist = math.hypot(dx, dy)
        self.start_time_ns = self.get_clock().now().nanoseconds
        self.duration_ns = max(1, int((xy_dist / max(0.01, speed_xy)) * 1e9))
        self.state_changed = False

    def _yaw_to_target_deg(self, current_xy: dict, target_wp: dict) -> float:
        """Return yaw in DEGREES for vehicle_command.publish_position_setpoint."""
        return geometry_utils.get_attitude(current_xy, target_wp)

    def _interpolate_and_publish(self):
        now = self.get_clock().now().nanoseconds
        t = min(1.0, (now - self.start_time_ns) / float(self.duration_ns)) if self.duration_ns else 1.0
        x = self.start_wp['x'] + (self.goal_wp['x'] - self.start_wp['x']) * t
        y = self.start_wp['y'] + (self.goal_wp['y'] - self.start_wp['y']) * t
        z = self.start_wp['z'] + (self.goal_wp['z'] - self.start_wp['z']) * t
        # Yaw toward the target from current local position using geometry_utils
        if self.local_pos is not None:
            cur = {'x': self.local_pos.x, 'y': self.local_pos.y}
            yaw = self._yaw_to_target_deg(cur, self.goal_wp)
        else:
            yaw = 0.0
        self._publish_position_sp(x, y, z, yaw)
        return t

    def _within_tolerance(self, goal_wp: dict) -> bool:
        if self.local_pos is None:
            return False
        total_dist, dist_xy, dist_z = geometry_utils.get_distance_between_ned(
            self.local_pos.x, self.local_pos.y, self.local_pos.z,
            goal_wp['x'], goal_wp['y'], goal_wp['z'])
        xy_ok = dist_xy <= self.xy_tol
        z_ok = dist_z <= self.z_tol
        return xy_ok and z_ok

    # ---------- Main timer ----------
    def timer_cb(self):
        # Wait until we have EKF origin & params before arming
        if self.state == 'WAIT_ORIGINS':
            if self.origin_set and len(self.wp_list_raw) > 0:
                self.get_logger().info('Origin available. Engaging offboard & arming...')
                vehicle_command.engage_offboard_mode(self.cmd_pub, self.get_clock())
                vehicle_command.arm(self.cmd_pub, self.get_clock())
                self.state = 'WAIT_ARM'
                self.state_changed = True
            return

        # Wait until ARM latched (wp0) then build NED WPs
        if self.state == 'WAIT_ARM':
            if (self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                and self.local_pos is not None and not self.arm_set and self.origin_set):
                self.wp0_ned = {
                    'x': float(self.local_pos.x),
                    'y': float(self.local_pos.y),
                    'z': float(self.local_pos.z),
                }
                self.wp0_z = float(self.local_pos.z)
                self.arm_set = True
                self.get_logger().info(
                    f'ARM latched (timer): wp0_ned=({self.wp0_ned["x"]:.2f}, {self.wp0_ned["y"]:.2f}, {self.wp0_ned["z"]:.2f})')
            if self._build_ned_waypoints_if_ready():
                self.get_logger().info('Waypoints converted to NED. TAKEOFF...')
                self.state = 'TAKEOFF'
                self.state_changed = True
            return

        # TAKEOFF: vertical climb at wp0 (x0,y0) to takeoff altitude
        if self.state == 'TAKEOFF':
            if self.state_changed:
                start = {'x': float(self.local_pos.x), 'y': float(self.local_pos.y), 'z': float(self.local_pos.z)}
                goal = {'x': self.takeoff_ned['x'], 'y': self.takeoff_ned['y'], 'z': self.takeoff_ned['z']}
                self._leg_init(start, goal, speed_xy=max(0.5, self.speed_xy))
            t = self._interpolate_and_publish()
            # Height-only tolerance = z_tol (1 m)
            if geometry_utils.is_height_reached(self.local_pos.z, self.takeoff_ned['z'], self.z_tol) or t >= 1.0:
                self.get_logger().info('Takeoff altitude reached; proceed to WP1')
                self.state = 'WP_NAV'
                self.leg = 0
                self.state_changed = True
            return

        # WP legs: WP1..WP4 at constant speed, yaw toward target
        if self.state == 'WP_NAV':
            if self.leg < len(self.wps_ned):
                target = self.wps_ned[self.leg]
                if self.state_changed:
                    start = {'x': self.local_pos.x, 'y': self.local_pos.y, 'z': self.local_pos.z}
                    self._leg_init(start, target, speed_xy=self.speed_xy)
                t = self._interpolate_and_publish()
                if self._within_tolerance(target) or t >= 1.0:
                    self.leg += 1
                    self.state_changed = True
                    next_label = f'WP{self.leg+1}' if self.leg < len(self.wps_ned) else 'RTL'
                    self.get_logger().info(f'WP{self.leg} reached -> {next_label}')
            else:
                self.state = 'RTL'
                self.state_changed = True
            return

        # RTL to wp0 at takeoff altitude, then LAND
        if self.state == 'RTL':
            target = {'x': self.wp0_ned['x'], 'y': self.wp0_ned['y'], 'z': self.takeoff_ned['z']}
            if self.state_changed:
                start = {'x': self.local_pos.x, 'y': self.local_pos.y, 'z': self.local_pos.z}
                self._leg_init(start, target, speed_xy=self.speed_xy)
            t = self._interpolate_and_publish()
            if self._within_tolerance(target) or t >= 1.0:
                self.get_logger().info('At wp0 above, initiating LAND')
                self.state = 'LAND'
                self.state_changed = True
            return

        if self.state == 'LAND':
            # Hold above wp0 and send LAND command
            self._publish_position_sp(self.wp0_ned['x'], self.wp0_ned['y'], self.takeoff_ned['z'], 0.0)
            vehicle_command.land(self.cmd_pub, self.get_clock())
            return


def main(args=None):
    rclpy.init(args=args)
    node = OffboardWaypointNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
