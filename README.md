# online_adaptive_cbf

This repository contains the implementation of an online adaptive framework for Control Barrier Functions (CBFs) in input-constrained nonlinear systems. The algorithm dynamically adapts CBF parameters to optimize performance while ensuring safety, particularly for robotic navigation tasks. Please see our paper ["Learning to Refine Input Constrained Control Barrier Functions via Uncertainty-Aware Online Parameter Adaptation"]() for more details.


<div align="center">
<img src="https://github.com/user-attachments/assets/d047f394-008f-4eb2-9e79-3e2f7a074b47" width="700px">

<div align="center">

[[Homepage]](https://www.taekyung.me/online-adaptive-cbf)
[[Arxiv]](https://arxiv.org/abs/2409.14616)
[[Video]](https://youtu.be/255IUS1f6Lo)
[[Research Group]](https://dasc-lab.github.io/)

</div>
</div>


## Features

- Implementation of the Probabilistic Ensemble Neural Network ([PENN](https://github.com/tkkim-robot/online_adaptive_cbf/tree/main/nn_model/penn)) which offers parallelized inference without an outer for loop. The predicted output can be interpreted as a Gaussian Mixture Model (GMM). (see [Kim et al.](https://arxiv.org/abs/2305.12240))
- Measurement of [closed-form epistemic uncertainty](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/nn_model/penn/divergence/utility.py) from the PENN model's predictions. (see [Kim et al.](https://arxiv.org/abs/2305.12240))
- Integration with the [`safe_control`](https://github.com/tkkim-robot/safe_control) repository for simulating robotic navigation, offering various robot dynamics, controllers, and RGB-D type sensor simulation.
- Implementation of the Online Adaptive ICCBF, adapting ICCBF parameters online based on the robot's current state and nearby environment.


## Installation
To install this project, follow these steps:

1. Clone the repository:
   ```bash
   git --recursive clone https://github.com/tkkim-robot/online_adaptive_cbf.git
   cd online_adaptive_cbf
   ```

   If you've already cloned the repository without the --recursive flag, you can initialize and update the submodules with:
   ```bash
   submodule update --init --recursive
   ```

2. (Optional) Create and activate a virtual environment

3. Install the package and its dependencies:
   ```bash
   python -m pip install -e .
   ```
   Or, install packages manually (see [`setup.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/setup.py)).


## Getting Started

Familiarize with APIs and examples with the scripts in [`online_adaptive_cbf.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/online_adaptive_cbf.py)

### Basic Example
You can run our test example by:

```bash
python online_adaptive_cbf.py
```

The MPC-CBF framework is implemented in our [`safe_control`](https://github.com/tkkim-robot/safe_control) repository. It imports `LocalTrackingController` class and uses the `mpc_cbf` implementation:
```python
from safe_control.tracking import LocalTrackingController
controller = LocalTrackingController(x_init, robot_spec,
                                control_type='mpc_cbf')
```

Then, it uses `OnlineCBFAdapter` to adapt the CBF parameters online.

```python
online_cbf_adapter = OnlineCBFAdapter(nn_model, scaler)

for _ in range(int(tf / self.dt)):
   ret = controller.control_step()
   controller.draw_plot()

   best_gamma0, best_gamma1 = online_cbf_adapter.cbf_param_adaptation(controller)
   controller.pos_controller.cbf_param['alpha1'] = best_gamma0
   controller.pos_controller.cbf_param['alpha2'] = best_gamma1    
```

You can also test with compared methods:

- Fixed CBF parameters:
   - `mpc_cbf` with conservatie fixed parameters: Set lower values to CBF parameters. (* MPC-CBF: An MPC controller using discrete-time CBF, ref: [[1]](https://ieeexplore.ieee.org/document/9483029))
      ```python
         controller.pos_controller.cbf_param['alpha1'] = {low value}
         controller.pos_controller.cbf_param['alpha2'] = {low value}
      ```
   - `mpc_cbf` with aggressive fixed parameters: Similarly, set higher values to CBF parameters.
- Adaptive parameter methods:
   - `optimal_decay_cbf_qp`: A modified CBF-QP for point-wise feasibility guarantee (ref: [[2]](https://ieeexplore.ieee.org/document/9482626))
      ```python
      controller = LocalTrackingController(..., control_type='optimal_decay_cbf_qp')
      ```
   - `optimal_decay_mpc_cbf`: The same technique applied to MPC-CBF (ref: [[3]](https://ieeexplore.ieee.org/document/9683174))
      ```python
      controller = LocalTrackingController(..., control_type='optimal_decay_mpc_cbf')
      ```


The sample results from the basic example:

|     MPC-CBF w/ low parameters            |       MPC-CBF w/ high parameters     |
| :------------------: | :--------------------------: |
|  <img src="https://github.com/user-attachments/assets/6a67bf2d-0c0f-437f-8fc0-8d21511b9ab6"  height="170px"> | <img src="https://github.com/user-attachments/assets/8151d102-6fbe-4a93-8967-7004c8e0b2cb"  height="170px"> |

|     Optimal Decay CBF-QP  |       Optimal Decay MPC-CBF    |
| :---------------------: | :----------------------------: |
|  <img src="https://github.com/user-attachments/assets/e43f72bc-475a-403d-bac8-41a077acdaf1"  height="170px"> | <img src="https://github.com/user-attachments/assets/ae2ecb58-254b-4334-84d6-8c52508c9973"  height="170px"> |


|     Ours (Online Adaptive MPC-ICCBF)      |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/5d5806c1-31a9-42fb-806f-04ece91d54ba"  height="170px"> |

The green point is the goal location, and the gray circles are the obstacles that are known a priori.

## Module Breakdown

### Safety Loss Density Function

The [`safety loss density function`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/safety_loss_function.py) is designed to quantify the collision risk between the robot and obstacles. This safety loss is computed based on the robot's state and the obstacles' locations.

|    Mean Predicted Risk Level    |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/9d8d2e21-ab83-4445-9cc4-de09029550b2"  height="350px"> |

### Data Generation

You can use [`data_generation.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/data_generation.py) to collect training dataset. It will store `risk_level` and `deadlock_time` as the ground truth. 

The `risk_level` refers to the maximum safety loss value recorded during the navigation (see the illustration below).

|    Safety Loss during Navigation     |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/591813c0-15e3-4857-8949-eef009a2697a"  height="250px"> |


### PENN Prediction

To train the PENN model, use the script [`penn/train_data.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/nn_model/train_data.py). An example of the prediction results after training is shown below: 

|    Mean Predicted Risk Level    |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/611dea04-93df-4635-97eb-34b93a5850ad"  height="350px"> |

You can observe that the predicted risk level becomes higher as the CBF parameter increases, the distance to the obstacle decreases, the velocity increases, and the relative angle to the obstacle becomes smaller.

### Distributionally Robust CVaR

<p align="left">
  <img src="https://github.com/user-attachments/assets/b6da3010-dcac-47a9-ad63-34a03df8e29a" height="200px">
</p>

Please refer to our repository [`cvar-gmm-filter`](https://github.com/signalkee/cvar-gmm-filter/tree/7405e05f7455f320b2c7b0ae72cef31a82d4a4f8) for more details.

### Visualize Prediction Results for CBF Parameters of Interest

[`test_plot.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/test_plot.py) provides an online plotting tool to visualize the predicted GMM distribution of the candidate CBF parameters. Here is the example of visualizing the predicted `risk_level` with three candidates, without adapting the paremeters.


|    Single Obstacle         |    Multiple Obstacles    |
| :------------------: | :--------------------------: |
|  <img src="https://github.com/user-attachments/assets/3a883e17-bda5-4719-a6ac-92104e0209ff"  height="250px"> | <img src="https://github.com/user-attachments/assets/f62659bd-9d4d-4ff1-9b9b-9f24f0dd85c7"  height="250px"> |



## Citing

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2025learning, 
    author    = {Kim, Taekyung and Kee, Robin Inho and Panagou, Dimitra},
    title     = {Learning to Refine Input Constrained Control Barrier Functions via Uncertainty-Aware Online Parameter Adaptation}, 
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    shorttitle = {Online-Adaptive-CBF},
    year      = {2025}
}
```

## Related Works

Here are some related projects/codes that you might be interested:

- [Visibility-Aware RRT*](https://github.com/tkkim-robot/visibility-rrt): Safety-critical Global Path Planning (GPP) using Visibility Control Barrier Functions

- [UGV Experiments with ROS2](https://github.com/tkkim-robot/px4_ugv_exp): Environmental setup for rovers using PX4, ros2 humble, Vicon MoCap, and NVIDIA VSLAM + NvBlox
