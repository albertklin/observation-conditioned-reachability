"""QP-based safety filter using the OCR value network."""

from typing import Tuple

import numpy as np
import torch
import cvxpy as cp

from utils.dynamics import Dubins3D


class SafetyFilter:
    """QP-based least-restrictive safety filter using OCR value network.

    The safety filter modifies nominal control commands to ensure safety,
    while staying as close as possible to the nominal command. It uses
    a learned value function V(state, lidar, disturbance_bounds) that
    predicts the minimum time-to-reach the obstacle set.

    The filter solves:
        min_{u} ||u - u_nom||^2 + slack_weight * slack^2
        s.t.  ∇V · f(x, u) >= -slack + disturbance_margin
              u_min <= u <= u_max
              slack >= 0

    where the disturbance margin accounts for worst-case disturbances.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dynamics: Dubins3D,
        control_min: np.ndarray,
        control_max: np.ndarray,
        filter_threshold: float = 0.35,
        calibration_value_adjustment: float = -0.5,
        slack_coeff: float = 1e3,
        device: torch.device = None,
    ):
        """Initialize the safety filter.

        Args:
            model: Trained OCR value network (LiDARValueNN).
            dynamics: Robot dynamics model.
            control_min: Minimum control bounds [v_x_min, v_yaw_min].
            control_max: Maximum control bounds [v_x_max, v_yaw_max].
            filter_threshold: Value above which no filtering is applied.
            calibration_value_adjustment: Offset added to value predictions
                for conservative estimation (typically negative, from calibrate_value_network.py).
            slack_coeff: Weight on slack variable in QP.
            device: Torch device for inference (default: auto-detect).
        """
        self.model = model
        self.dynamics = dynamics
        self.control_min = np.asarray(control_min)
        self.control_max = np.asarray(control_max)
        self.filter_threshold = filter_threshold
        self.calibration_value_adjustment = calibration_value_adjustment
        self.slack_coeff = slack_coeff
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def filter(
        self,
        state: np.ndarray,
        inp_lidar: np.ndarray,
        dst: np.ndarray,
        nom_v_x: float,
        nom_v_yaw: float,
        lidar_state: np.ndarray = None,
    ) -> Tuple[float, np.ndarray]:
        """Applies safety filtering to a nominal command.

        Args:
            state: Current robot state [x, y, theta].
            inp_lidar: Processed LiDAR scan, shape (num_rays,).
            dst: Disturbance bounds [dst_dxdy_max, dst_dth_max].
            nom_v_x: Nominal forward velocity.
            nom_v_yaw: Nominal yaw rate.
            lidar_state: Robot state when LiDAR was captured [x, y, theta].
                If None, uses current state (evaluates at origin in ego frame).

        Returns:
            Tuple of (predicted_value, filtered_control) where
            filtered_control is [v_x, v_yaw].
        """
        if lidar_state is None:
            lidar_state = state

        # Compute value and gradient
        pred_value, pred_grad = self._compute_value_and_gradient(
            state, lidar_state, inp_lidar, dst
        )

        # Apply least-restrictive filter
        filtered_control = self._solve_qp(
            pred_value, state[2], pred_grad, nom_v_x, nom_v_yaw, dst
        )

        return pred_value, filtered_control

    def _compute_value_and_gradient(
        self,
        state: np.ndarray,
        lidar_state: np.ndarray,
        inp_lidar: np.ndarray,
        dst: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Computes value function and gradient at current state.

        The value network expects state in the LiDAR frame (where LiDAR was
        captured). This computes the relative state: current_state - lidar_state
        rotated into the lidar frame. This accounts for robot motion between
        when the LiDAR scan was captured and the current control timestep.

        Args:
            state: Current robot state [x, y, theta].
            lidar_state: Robot state when LiDAR was captured [x, y, theta].
            inp_lidar: Processed LiDAR scan.
            dst: Disturbance bounds.

        Returns:
            Tuple of (value, gradient) where gradient is [dV/dx, dV/dy, dV/dθ]
            in world frame.
        """
        # Compute relative state in lidar/ego frame
        # This is the current position relative to where the robot was when
        # the LiDAR scan was captured, rotated into the lidar frame
        cth, sth = np.cos(lidar_state[2]), np.sin(lidar_state[2])
        rot = np.array([
            [cth, sth],
            [-sth, cth],
        ])
        rel_state = state.copy()
        rel_state[:2] = np.matmul(rot, (state[:2] - lidar_state[:2])[:, np.newaxis]).squeeze(axis=-1)
        rel_state[2] = state[2] - lidar_state[2]
        rel_state = self.dynamics.wrap_states(rel_state[np.newaxis])[0]

        # Build input tensor on the appropriate device
        inputs = torch.as_tensor(
            np.concatenate((rel_state, dst, inp_lidar), axis=-1)[np.newaxis],
            dtype=torch.float32,
            device=self.device,
        )

        # Enable gradient computation for state variables
        states = inputs[:, :3].detach().clone().requires_grad_(True)
        inputs = torch.cat([states, inputs[:, 3:]], dim=1)

        # Forward pass
        pred_values = self.model.forward(inputs) + self.calibration_value_adjustment
        pred_grad = torch.autograd.grad(
            pred_values.unsqueeze(-1), states,
            torch.ones_like(pred_values.unsqueeze(-1)),
            create_graph=False
        )[0][0].detach().cpu().numpy()

        # Rotate gradient to world frame
        pred_grad[:2] = np.matmul(rot.T, pred_grad[:2, np.newaxis]).squeeze(-1)

        return pred_values.item(), pred_grad

    def _solve_qp(
        self,
        val: float,
        th: float,
        grad: np.ndarray,
        nom_v_x: float,
        nom_v_yaw: float,
        dst: np.ndarray,
    ) -> np.ndarray:
        """Solves the safety QP for least-restrictive control.

        Args:
            val: Current value function.
            th: Current heading angle.
            grad: Value gradient [dV/dx, dV/dy, dV/dθ] in world frame.
            nom_v_x: Nominal forward velocity.
            nom_v_yaw: Nominal yaw rate.
            dst: Disturbance bounds [dst_dxdy_max, dst_dth_max].

        Returns:
            Filtered control [v_x, v_yaw].
        """
        dvdx, dvdy, dvdth = grad

        if val > self.filter_threshold:
            return np.array([nom_v_x, nom_v_yaw])

        sth, cth = np.sin(th), np.cos(th)
        try:
            ctrl = cp.Variable(2)
            slack = cp.Variable(1)
            prob = cp.Problem(
                cp.Minimize(
                    cp.sum_squares(ctrl - np.array([nom_v_x, nom_v_yaw]))
                    + self.slack_coeff * cp.square(slack)
                ),
                [
                    cp.sum(cp.multiply(ctrl, np.array([dvdx*cth + dvdy*sth, dvdth])))
                    >= -slack + dst[0]*np.linalg.norm([dvdx, dvdy]) + dst[1]*np.abs(dvdth),
                    ctrl >= self.control_min,
                    ctrl <= self.control_max,
                    slack >= 0,
                ],
            )
            prob.solve()
            if prob.status != 'optimal':
                raise Exception(f'prob.status: {prob.status}')
            return ctrl.value
        except Exception as e:
            # QP failed - use fallback
            print(f'QP failed: {e}')
            v_x = self.control_min[0] if (dvdx*cth + dvdy*sth) < 0 else self.control_max[0]
            v_yaw = self.control_min[1] if dvdth < 0 else self.control_max[1]
            return np.array([v_x, v_yaw])
