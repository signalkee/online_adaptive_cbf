import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'safe_control'))
sys.path.append(os.path.join(project_root, 'cvar_gmm_filter'))

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from scipy.stats import norm
from torch_geometric.data import Batch
from sklearn.preprocessing import MinMaxScaler
from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController
from nn_model.penn.gat import GATModule
from nn_model.penn.nn_iccbf_predict import ProbabilisticEnsembleNN
from nn_model.penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT
from cvar_gmm_filter.distributionally_robust_cvar import DistributionallyRobustCVaR
from online_cbf_config import ALL_DEFAULTS, ADAPTIVE_MODELS

class OnlineCBFAdapter:
    def __init__(self, model_name, scaler_name=None, d_min=0.075, step_size=0.05,
                 epistemic_threshold=0.3, lower_bound=0.01, upper_bound=1.0,
                 robot_model=None, use_gat=False):
        """
        Initialize the adaptive CBF parameter selector
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        print(f"Using device {self.device} for OnlineCBFAdapter")
        self.robot_model = robot_model
        if self.robot_model == 'Quad2D': #TODO: make state dic
            self.extra_state = 1
        else:
            self.extra_state = 0

        self.use_gat = use_gat
        if self.use_gat:
            self.n_states = 18
        else:
            self.n_states = 6 + self.extra_state

        self.gat_module = None
        if self.use_gat:
            self.gat_module = GATModule().to(self.device)
            self.penn = ProbabilisticEnsembleGAT(self.gat_module, device=self.device)
        else:
            self.penn = ProbabilisticEnsembleNN(n_states=self.n_states, device=self.device)
            if scaler_name:
                self.penn.load_scaler(scaler_name)

        self.penn.load_model(model_name)
        self.lower_bound = lower_bound  # Lower bound for CBF parameter sampling, Conservative
        self.upper_bound = upper_bound  # Upper bound for CBF parameter sampling, Aggressive
        self.d_min = d_min  # Closest allowable distance to obstacles
        self.step_size = step_size  # Step size for sampling caldidate CBF parameters
        self.epistemic_threshold = epistemic_threshold  # Threshold for filtering predictions based on epistemic uncertainty

    def sample_cbf_parameters(self, current_gamma0, current_gamma1):
        '''
        Sample CBF parameters (gamma0 and gamma1) within a specified range
        '''
        gamma0_range = np.arange(max(self.lower_bound, current_gamma0 - 2.5), min(self.upper_bound, current_gamma0 + 2.5 + self.step_size), self.step_size)
        gamma1_range = np.arange(max(self.lower_bound, current_gamma1 - 2.5), min(self.upper_bound, current_gamma1 + 2.5 + self.step_size), self.step_size)
        return gamma0_range, gamma1_range

    def get_rel_state_wt_obs(self, tracking_controller):
        """
        Get the relative state of the robot with respect to the nearest obstacle
        """
        robot_pos = tracking_controller.robot.X[:2, 0].flatten()
        robot_theta = tracking_controller.robot.X[2, 0]
        robot_radius = tracking_controller.robot.robot_radius
        try:
            near_obs = tracking_controller.nearest_obs.flatten()
        except:
            near_obs = [100, 100, 0.2]  # Default obstacle in case of no nearby obstacle
        
        # Calculate distance, velocity, and relative angle with the obstacle
        if self.robot_model == 'VTOL2D':
            distance = np.linalg.norm(robot_pos - near_obs[:2]) - robot_radius - near_obs[2]
            #distance = np.linalg.norm(robot_pos - near_obs[:2]) - 0.45 + robot_radius + near_obs[2] # correct the distance for mistake in the training data
            # abuse variable name: it's just pitch angle in this scenario (since all obs are fixed)
            delta_theta = robot_theta 
        else:
            distance = np.linalg.norm(robot_pos - near_obs[:2]) - robot_radius - near_obs[2]
            #distance = np.linalg.norm(robot_pos - near_obs[:2]) - 0.45 + robot_radius + near_obs[2]
            delta_theta = np.arctan2(near_obs[1] - robot_pos[1], near_obs[0] - robot_pos[0]) - robot_theta
            delta_theta = ((delta_theta + np.pi) % (2 * np.pi)) - np.pi  
        gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
        gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']

        # If Quad2D => velocity_x, velocity_z
        if self.robot_model == 'Quad2D':
            velocity_x = tracking_controller.robot.X[3, 0]
            velocity_z = tracking_controller.robot.X[4, 0]
            return [distance, velocity_x, velocity_z, delta_theta, gamma0, gamma1]
        else:
            # for vtol, also put x_vel only in this particular scenario (same setting for training)            
            # 2D ground => velocity is single scalar
            # for vtol, also put x_vel only in this particular scenario (same setting for training)
            velocity = tracking_controller.robot.X[3, 0]
            return [distance, velocity, delta_theta, gamma0, gamma1]

    def predict_with_penn(self, current_state, gamma0_range, gamma1_range):
        """
        Predict safety loss, deadlock time, and epistemic uncertainty using PENN (MLP-based)
        """
        g0_grid, g1_grid = np.meshgrid(gamma0_range, gamma1_range, indexing='ij')
        gamma_flat = np.stack([g0_grid.flatten(), g1_grid.flatten()], axis=1)  # (N, 2)
        num_samples = gamma_flat.shape[0]

        state_repeated = np.tile(current_state, (num_samples, 1))  # (N, D)
        state_repeated[:, 3 + self.extra_state] = gamma_flat[:, 0]
        state_repeated[:, 4 + self.extra_state] = gamma_flat[:, 1]

        # Predict using vectorized PENN
        y_pred_safety_loss, y_pred_deadlock_time, epistemic_uncertainty = self.penn.predict(state_repeated)

        # Repackage predictions
        predictions = [
            (gamma_flat[i, 0], gamma_flat[i, 1], y_pred_safety_loss[i], y_pred_deadlock_time[i][0], epistemic_uncertainty[i])
            for i in range(num_samples)
        ]
        return predictions    
    
    def build_graph_from_env(self, tracking_controller):
        """
        Build a PyG graph from the current environment
        """
        rx, ry = tracking_controller.robot.X[0, 0], tracking_controller.robot.X[1, 0]
        rtheta = tracking_controller.robot.X[2, 0]

        # Convert heading+velocity to vx,vy if ground robot
        if self.robot_model == 'Quad2D':
            vx = tracking_controller.robot.X[3, 0]
            vz = tracking_controller.robot.X[4, 0]
            robot_state = [rx, ry, vx, vz]
        else:
            vel = tracking_controller.robot.X[3, 0]
            vx = vel * np.cos(rtheta)
            vy = vel * np.sin(rtheta)
            robot_state = [rx, ry, vx, vy]

        obstacles = tracking_controller.nearest_multi_obs
        if isinstance(obstacles, np.ndarray):
            if obstacles.size == 0:
                obstacles = [[100., 100., 0.2]]
        else:
            if not obstacles:
                obstacles = [[100., 100., 0.2]]

        final_waypoint = tracking_controller.waypoints[-1]
        goal = [final_waypoint[0], final_waypoint[1]]

        # Build the graph using the GATModule
        gdata = self.gat_module.create_graph(
            robot=robot_state,
            obstacles=obstacles,
            goal=goal,
            deadlock=0.0,  # placeholders
            risk=0.0
        )
        return gdata
    
    def predict_with_gat_penn(self, tracking_controller, gamma0_range, gamma1_range):
        """
        Predict safety loss, deadlock time, and epistemic uncertainty 
        using the Probabilistic Ensemble Neural Network
        """
        base_graph = self.build_graph_from_env(tracking_controller)

        # Generate all gamma combinations
        g0_grid, g1_grid = torch.meshgrid(
            torch.tensor(gamma0_range, dtype=torch.float32),
            torch.tensor(gamma1_range, dtype=torch.float32),
            indexing='ij'
        )
        gamma_comb = torch.stack([g0_grid.flatten(), g1_grid.flatten()], dim=1).to(self.device)  # (N, 2)
        num_samples = gamma_comb.shape[0]

        graph_list = [base_graph.clone() for _ in range(num_samples)]
        for i in range(num_samples):
            graph_list[i].gamma = gamma_comb[i].unsqueeze(0)  # Shape (1, 2)
        batched_graph = Batch.from_data_list(graph_list).to(self.device)

        # Predict with vectorized PENN
        y_pred_safety_list, y_pred_deadlock_list, div_list = self.penn.predict([batched_graph])
        predictions = []
        for i, (g0, g1) in enumerate(gamma_comb.cpu().numpy()):
            predictions.append((g0, g1, y_pred_safety_list[i], y_pred_deadlock_list[i], div_list[i]))

        return predictions

    def filter_by_epistemic_uncertainty(self, predictions):
        '''
        Filter predictions based on epistemic uncertainty
        We employ Jensen-Renyi Divergence (JRD) with quadratic Renyi entropy, which has a closed-form expression of the divergence of a GMM
        If the JRD D(X) of the prediction of a given input X is greater than the predefined threshold, it is deemed to be out-of-distribution
        '''        
        if not predictions:
            return []
        epi = np.asarray([p[4] for p in predictions], dtype=np.float32)          # (N,)
        # If all uncertainties are high, return an empty list
        if np.all(epi > 5.0):
            return []
        epi_norm = (epi - epi.min()) / (epi.max() - epi.min() + 1e-8)
        keep_mask = epi_norm <= self.epistemic_threshold                         # (N,) bool
        return [pred for pred, keep in zip(predictions, keep_mask) if keep]

    def calculate_cvar_boundary(self):
        '''
        Calculate the boundary where that the class K functions are locally valid
        '''
        lambda_1 = 0.4  # Same value used from the data generation step
        beta_1 = 100.0  # Same value used from the data generation step
        d_min = self.d_min
        cvar_boundary = lambda_1 / (beta_1 * d_min**2 + 1)
        return cvar_boundary

    def filter_by_aleatoric_uncertainty(self, filtered_predictions):
        '''
        Filter predictions (GMM distribution due to ensemble predictions) based on aleatoric uncertainty 
        using the distributionally robust Conditional Value at Risk (CVaR) boundary
        '''
        if not filtered_predictions:
            return []

        # y_pred_safety_loss[i] == list_of_ensembles  => [ [mu, var], ... ]
        N, E = len(filtered_predictions), self.penn.n_ensemble
        mu_mat   = np.zeros((N, E), dtype=np.float64)
        sig2_mat = np.zeros((N, E), dtype=np.float64)

        for i, pred in enumerate(filtered_predictions):
            ens = pred[2]  # y_pred_safety_loss  => list[[mu,var],...]
            mu_mat[i]   = [e[0] for e in ens]
            sig2_mat[i] = [e[1] for e in ens]

        boundary  = self.calculate_cvar_boundary()
        keep_mask = DistributionallyRobustCVaR.batch_within_boundary(mu_mat, sig2_mat, boundary, alpha=0.99)

        return [pred for pred, keep in zip(filtered_predictions, keep_mask) if keep]

    def select_best_parameters(self, final_predictions, tracking_controller):
        '''
        Select the best CBF parameters based on filtered predictions.
        '''
        # If no predictions were selected, gradually decrease the parameter
        if not final_predictions:
            current_gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
            current_gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']
            gamma0 = max(self.lower_bound, current_gamma0 - self.step_size)
            gamma1 = max(self.lower_bound, current_gamma1 - self.step_size)
            return gamma0, gamma1
        
        # min_deadlock_time = min(final_predictions, key=lambda x: x[3])[3]
        # print(final_predictions)
        # best_predictions = [pred for pred in final_predictions if pred[3][0] < 1e-3]
        # If no predictions under 1e-3, use the minimum deadlock time
        # if not best_predictions:
        #     best_predictions = [pred for pred in final_predictions if pred[3] == min_deadlock_time]
        # # If there are multiple best predictions, use harmonic mean to select the best one
        # if len(best_predictions) != 1:
        #     best_prediction = max(best_predictions, key=lambda x: 2 * (x[0] * x[1]) / (x[0] + x[1]) if (x[0] + x[1]) != 0 else 0)
        #     return best_prediction[0], best_prediction[1]
        # return best_predictions[0][0], best_predictions[0][1]

        # Pick the best prediction by harmonic mean of (gamma0, gamma1).
        best_prediction = max(
            final_predictions, 
            key=lambda x: 2.0 * (x[0] * x[1]) / (x[0] + x[1]) if (x[0] + x[1]) != 0 else 0.0
        )

        return best_prediction[0], best_prediction[1]

    def cbf_param_adaptation(self, tracking_controller):
        '''
        Perform adaptive CBF parameter selection based on the prediction from the PENN model 
        which is both confident and satisfies the local validity condition
        '''
        current_state = self.get_rel_state_wt_obs(tracking_controller)
        gamma0_range, gamma1_range = self.sample_cbf_parameters(current_state[3+self.extra_state], current_state[4+self.extra_state])
        
        if self.use_gat:
            predictions = self.predict_with_gat_penn(tracking_controller, gamma0_range, gamma1_range)
        else:
            predictions = self.predict_with_penn(current_state, gamma0_range, gamma1_range)
        
        filtered_predictions = self.filter_by_epistemic_uncertainty(predictions)
        final_predictions = self.filter_by_aleatoric_uncertainty(filtered_predictions)
        best_gamma0, best_gamma1 = self.select_best_parameters(final_predictions, tracking_controller)

        if best_gamma0 is not None and best_gamma1 is not None:
            print(f"CBF parameters updated to: {best_gamma0:.2f}, {best_gamma1:.2f}"
                  f" | Total predictions: {len(predictions)}"
                  f" | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic"
                  f" | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric")
        else:
            print(f"CBF parameters updated to: NONE, NONE"
                  f" | Total predictions: {len(predictions)}"
                  f" | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic"
                  f" | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric")

        return best_gamma0, best_gamma1



def get_robot_spec_and_obs(robot_model):
    """
    Returns (robot_spec, default_obs) for a given robot_model.
    """
    if robot_model not in ALL_DEFAULTS:
        raise ValueError(f"Unknown robot_model '{robot_model}'")
    
    entry = ALL_DEFAULTS[robot_model]
    return entry["robot_spec"], entry["default_obs"]

def get_controller_defaults(robot_model, controller_name):
    """
    Returns (controller_type, gamma0, gamma1) for the given robot_model
    and high-level controller_name.
    """
    if robot_model not in ALL_DEFAULTS:
        raise ValueError(f"Unknown robot_model '{robot_model}'")
    
    controller_params = ALL_DEFAULTS[robot_model]["controller_params"].get(controller_name)
    if controller_params is None:
        raise ValueError(f"Unknown controller_name '{controller_name}' for robot_model '{robot_model}'")
    
    return (
        controller_params["type"],
        controller_params["gamma0"],
        controller_params["gamma1"]
    )

def get_env_defaults(robot_model):
    '''
    Returns environment configurations
    '''
    
    if robot_model not in ALL_DEFAULTS:
        raise ValueError(f"Unknown robot_model '{robot_model}'")

    entry = ALL_DEFAULTS[robot_model]

    env_width = entry.setdefault("env_width", 11.0)
    env_height = entry.setdefault("env_height", 3.8)

    return env_width, env_height

def get_online_cbf_adapter(robot_model, controller_name):
    """
    Returns an OnlineCBFAdapter instance for the given robot_model
    """
    if robot_model not in ADAPTIVE_MODELS:
        raise ValueError(f"No online adapter config found for '{robot_model}'")

    controller_subkey_map = {
        "Online Adaptive CBF-QP":  "online_cbf_qp",
        "Online Adaptive MPC-CBF MLP": "online_mpc_cbf_mlp",
        "Online Adaptive MPC-CBF GAT": "online_mpc_cbf_gat",
    }
    if controller_name == "Online Adaptive MPC-CBF GAT": 
        use_gat=True
    else:
        use_gat=False
    
    if controller_name not in controller_subkey_map:
        raise ValueError(f"Controller '{controller_name}' not recognized for online adaptation.")
    subkey = controller_subkey_map[controller_name]
    if subkey not in ADAPTIVE_MODELS[robot_model]:
        raise ValueError(f"No config for subkey '{subkey}' in '{robot_model}'")

    cfg = ADAPTIVE_MODELS[robot_model][subkey]
    return OnlineCBFAdapter(
        model_name=cfg["model_path"],
        scaler_name=cfg["scaler_path"],
        step_size=cfg["step_size"],
        lower_bound=cfg["lower_bound"],
        upper_bound=cfg["upper_bound"],
        epistemic_threshold=cfg.get("epistemic_threshold", 0.2),
        robot_model=robot_model,
        use_gat=use_gat
    )

def single_agent_simulation(velocity,
                            waypoints,
                            controller_name,
                            robot_model,
                            max_sim_time=30,
                            dt=0.05):
    """
    Run a single-agent trajectory simulation using the specified
    robot_model, controller strategy, initial velocity, and waypoints.
    """
    
    # Get the robot spec & default obstacles & default controller type & gamma values
    robot_spec, default_obs = get_robot_spec_and_obs(robot_model)
    ctrl_type, gamma0, gamma1 = get_controller_defaults(robot_model, controller_name)
    env_width, env_height = get_env_defaults(robot_model)

    print(robot_spec, default_obs)
    print(ctrl_type, gamma0, gamma1)

    if default_obs.shape[1] != 5:
        default_obs = np.hstack((default_obs, np.zeros((default_obs.shape[0], 2)))) # Set static obs velocity 0.0 at (5, 5)
    

    # Set initial state
    if robot_model == "Quad2D":
        # velocity should be [vx, vz] for Quad2D
        x_init = np.append(waypoints[0], [velocity[0], velocity[1], 0])
    elif robot_model == "VTOL2D":
        x_init = np.hstack((2.0, 10.0, 0.0, velocity, 0.0, 0.0))
        plt.rcParams['figure.figsize'] = [12, 5]
    else:
        # velocity is a single scalar for 2D ground vehicles
        x_init = np.append(waypoints[0], velocity)

    # Set plotting and environment
    plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=default_obs)
    ax, fig = plot_handler.plot_grid("")
    env_handler = env.Env()

    # Create the tracking controller
    tracking_controller = LocalTrackingController(
        x_init,
        robot_spec,
        control_type=ctrl_type,
        dt=dt,
        show_animation=False,
        save_animation=False,
        ax=ax,
        fig=fig,
        env=env_handler
    )

    # Initialize the CBF parameters
    tracking_controller.pos_controller.cbf_param['alpha1'] = gamma0
    tracking_controller.pos_controller.cbf_param['alpha2'] = gamma1

    # Load obstacles & set waypoints
    tracking_controller.obs = default_obs
    tracking_controller.set_waypoints(waypoints)

    # If controller is 'Online Adaptive', get adapter
    if controller_name in ['Online Adaptive CBF-QP', 'Online Adaptive MPC-CBF MLP', 'Online Adaptive MPC-CBF GAT']:
        online_cbf_adapter = get_online_cbf_adapter(robot_model, controller_name)
    else:
        online_cbf_adapter = None

    import csv
    # create a csv file to record the states, control inputs, and CBF parameters
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['states', 'control_inputs', 'alpha1', 'alpha2'])

    # Main simulation loop
    n_steps = int(max_sim_time / dt)
    import time
    for _ in range(n_steps):
        
        
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()

        # Check if we've reached the goal or collided
        if ret == -1:
            dist_to_goal = np.linalg.norm(tracking_controller.robot.X[:2, 0] - waypoints[-1][:2])
            if dist_to_goal < tracking_controller.reached_threshold:
                print("Goal point reached.")
            else:
                print("Collided.")
            break

        # Adapt the CBF parameters if using an online approach
        if online_cbf_adapter is not None:
            start = time.time()
            best_gamma0, best_gamma1 = online_cbf_adapter.cbf_param_adaptation(tracking_controller)
            if best_gamma0 is not None and best_gamma1 is not None:
                tracking_controller.pos_controller.cbf_param['alpha1'] = best_gamma0
                tracking_controller.pos_controller.cbf_param['alpha2'] = best_gamma1
            end = time.time()
            print(f"Time taken for pure adaptation step: {end - start:.4f} seconds")

        # get states of the robot
        robot_state = tracking_controller.robot.X[:,0].flatten()
        control_input = tracking_controller.get_control_input().flatten()
        # print(f"Robot state: {robot_state}")
        # print(f"Control input: {control_input}")

        # append the states, control inputs, and CBF parameters by appending to csv
        with open('output.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(np.append(robot_state, np.append(control_input, [tracking_controller.pos_controller.cbf_param['alpha1'], tracking_controller.pos_controller.cbf_param['alpha2']])))

    tracking_controller.export_video()
    plt.ioff()
    plt.close()


if __name__ == "__main__":
    controller_list = [
        "MPC-CBF low fixed param",
        "MPC-CBF high fixed param",
        "Optimal Decay CBF-QP",
        "Optimal Decay MPC-CBF",
        "Online Adaptive CBF-QP",
        "Online Adaptive MPC-CBF MLP",
        "Online Adaptive MPC-CBF GAT",
    ]
    robot_model_list = [
        "DynamicUnicycle2D",
        "KinematicBicycle2D",
        "Quad2D",
        "VTOL2D"
    ]

    # Pick a specific controller and robot model
    controller_name = controller_list[-1]   
    robot_model = robot_model_list[0]       
    
    # Define waypoints for the simulation
    if robot_model == "VTOL2D":
        waypoints = np.array([
                    [70, 10],
                    [70, 0.5]
                ], dtype=np.float64)
    else:
        waypoints = np.array([
                    [0.75, 2.0, 0.01],
                    [10.0, 1.5, 0.0]
                ], dtype=np.float64)

    # For ground vehicles, velocity is a single scalar
    if robot_model == "Quad2D":
        init_vel = [0.4, 0.2]
    elif robot_model == "VTOL2D":
        init_vel = 20.0
    else:
        init_vel = 0.4

    # Run the simulation
    single_agent_simulation(init_vel, waypoints, controller_name, robot_model)

