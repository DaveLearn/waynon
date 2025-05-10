# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from pickletools import optimize
from typing import Dict, List, Optional, Tuple

import esper
import numpy as np
import symforce

from waynon.components.optimizable import Optimizable
symforce.set_epsilon_to_symbol()
import sym
import sym.ops
import symforce.opt
import symforce.opt.factor
import symforce.opt.optimizer
import symforce.symbolic as sf
from symforce.opt.factor import Factor
from symforce.opt.noise_models import DiagonalNoiseModel
from symforce.opt.optimizer import Optimizer
from symforce.values import Values

from waynon.components.scene_utils import (get_world_id, is_dynamic,
                                           rotate_around_x)
from waynon.components.tree_utils import *

symforce.set_log_level("WARNING")


def to_sym_pose(X: np.ndarray, compiled=False):
    from scipy.spatial.transform import Rotation as R

    q = R.from_matrix(X[:3, :3]).as_quat()
    t = X[:3, 3]
    if not compiled:
        return sf.Pose3.from_storage([*q, *t])
    else:
        return sym.Pose3.from_storage([*q, *t])


def from_sym_pose(pose: sf.Pose3):
    from scipy.spatial.transform import Rotation as R

    q = pose.R.to_storage()
    t = pose.t
    r = R.from_quat(q)
    X = np.eye(4)
    X[:3, :3] = r.as_matrix()
    X[:3, 3] = t
    return X

@dataclass
class ConstKeys:
    @property
    def X_CV_C(self) -> str: return "X_CV_C"
    @property
    def epsilon(self) -> str: return "epsilon"

@dataclass
class CameraKeys:
    camera_id: int
    measurement_ids: List[Optional[int]] = field(default_factory= lambda: [None] * 4)

    @property
    def K(self) -> str: 
        return f"K_{self.camera_id}"
    
    def X_WC(self, idx: int) -> str: 
        assert idx > 0, f"idx must be greater than 0, got {idx}"
        if self.measurement_ids[idx-1] is not None:
            return f"X_WC_{idx}_{self.camera_id}_{self.measurement_ids[idx-1]}"
        else:
            return f"X_WC_{idx}_{self.camera_id}"
    
    def set_measurement_id(self, idx: int, measurement_id: Optional[int]):
        assert idx > 0, f"idx must be greater than 0, got {idx}"
        self.measurement_ids[idx-1] = measurement_id

@dataclass
class MarkerKeys:
    marker_id: int
    measurement_ids: List[Optional[int]] = field(default_factory= lambda: [None] * 3)

    def X_WM(self, idx: int) -> str: 
        assert idx > 0, f"idx must be greater than 0, got {idx}"
        if self.measurement_ids[idx-1] is not None:
            return f"X_WM_{idx}_{self.marker_id}_{self.measurement_ids[idx-1]}"
        else:
            return f"X_WM_{idx}_{self.marker_id}"

    def p_MP_M(self, point_num: int) -> str: return f"p_MP_{self.marker_id}_{point_num}"

    def set_measurement_id(self, idx: int, measurement_id: Optional[int]):
        assert idx > 0, f"idx must be greater than 0, got {idx}"
        self.measurement_ids[idx-1] = measurement_id


@dataclass
class MarkerMeasurementKeys:
    marker_measurement_id: int
    
    def point2D(self, point_num: int) -> str: return f"pixel_{self.marker_measurement_id}_{point_num}"



def eye_to_link_residual(
    point2D: sf.V2, # point on image u,v
    p_MP_M: sf.V3,  # point in marker frame
    K: sf.LinearCameraCal, # observing camera k
    X_WC_1: sf.Pose3, # camera pose chain transform 1
    X_WC_2: sf.Pose3, # camera pose chain transform 2
    X_WC_3: sf.Pose3, # camera pose chain transform 3
    X_CV_C: sf.Pose3, # camera pose to opencv convention
    X_WM_1: sf.Pose3, # marker pose chain transform 1
    X_WM_2: sf.Pose3, # marker pose chain transform 2
    X_WM_3: sf.Pose3, # marker pose chain transform 3
    epsilon: sf.Scalar,
) -> sf.V2:
    camera = sf.PosedCamera(pose=X_WC_1 * X_WC_2 * X_WC_3 * X_CV_C, calibration=K)
    p_MP_W = X_WM_1 * X_WM_2 * X_WM_3 * p_MP_M
    pixel, valid = camera.pixel_from_global_point(p_MP_W, epsilon=epsilon)
    if not valid:
        return None
    error = pixel - point2D
    return error


# returns list of (pose, optimize_entity_id, is_dynamic)
def get_tranform_chain(entity_id: int, max_length: int, measurement_id: int) -> List[Tuple[sf.Pose3, Optional[int], bool]]:
    from waynon.components.transform import Transform
    from waynon.components.robot import FrankaLink
    from waynon.components.joint_measurement import JointMeasurement
    from waynon.components.robot import Robot
    
    values = []
    optimize_target: List[Optional[int]] = []
    is_dynamic: List[bool] = []

    # get transformed parents on way to root
    world_id = get_world_id()
    current_id = entity_id
    parent_transformed_entities = []
    while current_id is not None and current_id != world_id:
        if esper.has_component(current_id, Transform) or esper.has_component(current_id, FrankaLink):
            parent_transformed_entities.append(current_id)

        current_id = get_node(current_id).parent_entity_id

    transform_list = parent_transformed_entities.copy()
    transform_list.reverse()


    last_was_constant = False

    for entity_id in transform_list:
        
        if esper.has_component(entity_id, FrankaLink):
            # use joint measurements to get transform.
            link = esper.component_for_entity(entity_id, FrankaLink)
            
            joint_measurement_id = find_descendant_with_component(measurement_id, JointMeasurement, predicate=lambda id, c: c.robot_id == link.robot_id)
            assert joint_measurement_id is not None, f"missing joint measurement for link {entity_id}"
            joint_measurement = esper.component_for_entity(joint_measurement_id, JointMeasurement)

            robot = esper.component_for_entity(link.robot_id, Robot)
            manager = robot.get_manager()
            assert manager is not None, f"missing manager for robot {link.robot_id}"

            q = joint_measurement.joint_values
            X_WR = manager.fk(q)[link.link_name]
            this_pose = to_sym_pose(X_WR)
            dynamic = True
        else:
            this_pose = to_sym_pose(esper.component_for_entity(entity_id, Transform).get_X_PT())
            dynamic = False

        want_optimize = False
        if esper.has_component(entity_id, Optimizable):
            want_optimize = esper.component_for_entity(entity_id, Optimizable).optimize
        
        if want_optimize or not last_was_constant:
            values.append(this_pose)
            optimize_target.append(entity_id if want_optimize else None)
            is_dynamic.append(dynamic)
        else:
            # combine previous pose if it also was constant
            if last_was_constant and len(values) >= 1:
                values[-1] = values[-1] * this_pose
                is_dynamic[-1] = is_dynamic[-1] or dynamic

        last_was_constant = not want_optimize

    assert len(values) > 0, f"couldn't find transform for entity {entity_id}"
    assert len(values) <= max_length, f"too many transforms for entity {entity_id} found {len(values)}"

    results = []
    for i in range(max_length):
        if i < len(values):
            results.append((values[i], optimize_target[i], is_dynamic[i]))
        else:
            results.append((sf.Pose3.identity(), None, False))

    return results



class FactorGraphSolver:
    def __init__(self):
        self.factors = []

    def add_camera(self, values: Values, optimized_keys_to_entity_id: Dict[str, int], camera_id: int, measurement_id: int) -> CameraKeys:
        from waynon.components.camera import PinholeCamera

        camera = esper.component_for_entity(camera_id, PinholeCamera)
        camera_keys = CameraKeys(camera_id)

        # K
        if not camera_keys.K in values:
            fl_x, fl_y = camera.fl_x, camera.fl_y
            cx, cy = camera.cx, camera.cy
            
            values[camera_keys.K] = sf.LinearCameraCal(
                focal_length=(fl_x, fl_y), principal_point=(cx, cy)
            )

        # get transforme chain camera
        transform_chain = get_tranform_chain(camera_id, max_length=3, measurement_id=measurement_id)
        for i, (pose, optimize_entity_id, is_dynamic) in enumerate(transform_chain, 1):
            camera_keys.set_measurement_id(i, measurement_id if is_dynamic else None)
            if not camera_keys.X_WC(i) in values:
                values[camera_keys.X_WC(i)] = pose
                if optimize_entity_id is not None:
                    optimized_keys_to_entity_id[camera_keys.X_WC(i)] = optimize_entity_id
        
        return camera_keys


    def add_marker_and_points(self, values: Values, optimized_keys_to_entity_id: Dict[str, int], marker_id: int, measurement_id: int) -> MarkerKeys:
        from waynon.components.aruco_marker import ArucoMarker
        from waynon.components.charuco_board import CharucoBoard

        marker_keys = MarkerKeys(marker_id)
        
        transform_chain = get_tranform_chain(marker_id, max_length=3, measurement_id=measurement_id)
        for i, (pose, optimize_entity_id, is_dynamic) in enumerate(transform_chain, 1):
            marker_keys.set_measurement_id(i, measurement_id if is_dynamic else None)
            if not marker_keys.X_WM(i) in values:
                values[marker_keys.X_WM(i)] = pose
                if optimize_entity_id is not None:
                    optimized_keys_to_entity_id[marker_keys.X_WM(i)] = optimize_entity_id

        # now add the points
        if esper.has_component(marker_id, ArucoMarker):
            aruco_marker: ArucoMarker = esper.component_for_entity(marker_id, ArucoMarker)
            p_MC = aruco_marker.get_P_MC()
            for i in range(p_MC.shape[0]):
                values[marker_keys.p_MP_M(i)] = p_MC[i]
                
        elif esper.has_component(marker_id, CharucoBoard):
            charuco_board: CharucoBoard = esper.component_for_entity(marker_id, CharucoBoard)
            p_MC = charuco_board.get_P_MC()
            for i in range(p_MC.shape[0]):
                values[marker_keys.p_MP_M(i)] = p_MC[i]

        else:
            raise ValueError(f"Unknown marker type for Entity: {marker_id}")
        
        return marker_keys




    def add_aruco_measurement(self, values: Values, optimized_keys_to_entity_id: Dict[str, int], aruco_measurement_id: int) -> List[Factor]:
        from waynon.components.aruco_measurement import ArucoMeasurement
        from waynon.components.measurement import Measurement
        
        aruco_measurement = esper.component_for_entity(aruco_measurement_id, ArucoMeasurement)
        valid = aruco_measurement.valid()
        if not valid:
            print(f"Skipping measurement {aruco_measurement_id}: {valid}")
            return []
        
        camera_id = aruco_measurement.camera_entity_id
        if esper.has_component(camera_id, Optimizable):
            opt = esper.component_for_entity(camera_id, Optimizable)
            if not opt.use_in_optimization:
                print(f"Skipping aruco measurement {aruco_measurement_id} as {camera_id} isn't marked for use measurements")
                return []
            
        marker_id = aruco_measurement.marker_entity_id
        if esper.has_component(marker_id, Optimizable):
            opt = esper.component_for_entity(marker_id, Optimizable)
            if not opt.use_in_optimization:
                print(f"Skipping aruco measurement {aruco_measurement_id} as {marker_id} isn't marked for use measurements")
                return []
       
        measurement_id = find_nearest_ancestor_with_component(aruco_measurement_id, Measurement)
        assert measurement_id is not None

        camera_keys = self.add_camera(values, optimized_keys_to_entity_id, camera_id, measurement_id)
        marker_keys = self.add_marker_and_points(values, optimized_keys_to_entity_id, marker_id, measurement_id)

        marker_measurement_keys = MarkerMeasurementKeys(aruco_measurement_id)
        const_keys = ConstKeys()

        # a factor per pixel
        factors = []
        for i, pixel in enumerate(aruco_measurement.pixels):
            values[marker_measurement_keys.point2D(i)] = sf.V2(pixel)

            factor = Factor(
                residual=eye_to_link_residual,
                keys=[
                    marker_measurement_keys.point2D(i),
                    marker_keys.p_MP_M(i),
                    camera_keys.K,
                    camera_keys.X_WC(1),
                    camera_keys.X_WC(2),
                    camera_keys.X_WC(3),
                    const_keys.X_CV_C,
                    marker_keys.X_WM(1),
                    marker_keys.X_WM(2),
                    marker_keys.X_WM(3),
                    const_keys.epsilon,
                ],
            )
            factors.append(factor)

        return factors


        
    def add_charuco_measurement(self, values: Values, optimized_keys_to_entity_id: Dict[str, int], charuco_measurement_id: int) -> List[Factor]:
        from waynon.components.charuco_board_measurement import CharucoBoardMeasurement
        from waynon.components.measurement import Measurement
        
        charuco_measurement = esper.component_for_entity(charuco_measurement_id, CharucoBoardMeasurement)
        valid = charuco_measurement.valid()
        if not valid:
            print(f"Skipping measurement {charuco_measurement_id}: {valid}")
            return []
        
        camera_id = charuco_measurement.camera_entity_id
        if esper.has_component(camera_id, Optimizable):
            opt = esper.component_for_entity(camera_id, Optimizable)
            if not opt.use_in_optimization:
                print(f"Skipping charuco measurement {charuco_measurement_id} as {camera_id} isn't marked for use measurements")
                return []
            

        marker_id = charuco_measurement.board_entity_id
        if esper.has_component(marker_id, Optimizable):
            opt = esper.component_for_entity(marker_id, Optimizable)
            if not opt.use_in_optimization:
                print(f"Skipping charuco measurement {charuco_measurement_id} as {marker_id} isn't marked for use measurements")
                return []



        measurement_id = find_nearest_ancestor_with_component(charuco_measurement_id, Measurement)
        assert measurement_id is not None

        camera_keys = self.add_camera(values, optimized_keys_to_entity_id, camera_id, measurement_id)
        marker_keys = self.add_marker_and_points(values, optimized_keys_to_entity_id, marker_id, measurement_id)

        marker_measurement_keys = MarkerMeasurementKeys(charuco_measurement_id)
        const_keys = ConstKeys()

        # a factor per pixel
        factors = []
        for i, pixel in zip(charuco_measurement.corner_ids, charuco_measurement.corner_pixels):
            values[marker_measurement_keys.point2D(i)] = sf.V2(pixel)

            factor = Factor(
                residual=eye_to_link_residual,
                keys=[
                    marker_measurement_keys.point2D(i),
                    marker_keys.p_MP_M(i),
                    camera_keys.K,
                    camera_keys.X_WC(1),
                    camera_keys.X_WC(2),
                    camera_keys.X_WC(3),
                    const_keys.X_CV_C,
                    marker_keys.X_WM(1),
                    marker_keys.X_WM(2),
                    marker_keys.X_WM(3),
                    const_keys.epsilon,
                ],
            )
            factors.append(factor)

        return factors

    async def run(self, factor_graph_id: int):
        from waynon.components.aruco_measurement import ArucoMeasurement
        from waynon.components.charuco_board_measurement import CharucoBoardMeasurement
        from waynon.components.camera import PinholeCamera
        from waynon.components.factor_graph import FactorGraph
        from waynon.components.joint_measurement import JointMeasurement
        from waynon.components.measurement import Measurement
        from waynon.components.optimizable import Optimizable
        from waynon.components.robot import FrankaLink, Robot
        from waynon.components.transform import Transform

        assert esper.entity_exists(factor_graph_id)
        assert esper.has_component(factor_graph_id, FactorGraph)
        factor_graph = esper.component_for_entity(factor_graph_id, FactorGraph)
        const_keys = ConstKeys()
        initial_values = Values({
            const_keys.epsilon: sf.numeric_epsilon,
            const_keys.X_CV_C: to_sym_pose(rotate_around_x(np.eye(4))),
        })

        optimized_keys_to_entity_id = {}

        factors = []

        for aruco_measurement_id, _ in esper.get_component(ArucoMeasurement):
            factors.extend(self.add_aruco_measurement(initial_values, optimized_keys_to_entity_id, aruco_measurement_id))

        for charuco_measurement_id, _ in esper.get_component(CharucoBoardMeasurement):
            factors.extend(self.add_charuco_measurement(initial_values, optimized_keys_to_entity_id, charuco_measurement_id))


        if len(factors) == 0:
            print("No factors found")
            return
        
        optimized_keys = list(optimized_keys_to_entity_id.keys())
        optimized_keys.sort() # ORDER MATTERS FOR SOME REASON. DO NOT REMOVE!!!!!!!!!!
        print(optimized_keys)

        optimizer = Optimizer(
            factors=[*factors],
            optimized_keys=optimized_keys,
            debug_stats=True,
            params=Optimizer.Params(
                verbose=factor_graph.verbose,
                iterations=factor_graph.iterations,
                early_exit_min_reduction=1e-10,
                initial_lambda=factor_graph.initial_lambda,
                enable_bold_updates=factor_graph.enable_bold_updates,
            ),
        )

        result = optimizer.optimize(initial_values)
        num_measurements = len(factors)
        print(result.status, result.error() / num_measurements)

        # dot_file = symforce.opt.factor.visualize_factors(factors, "factor_graph.dot")

        if result.status == Optimizer.Status.SUCCESS:
            optimized_values = result.optimized_values
            for key, entity_id in optimized_keys_to_entity_id.items():
                pose = from_sym_pose(optimized_values[key])
                #if esper.has_component(entity_id, PinholeCamera):
                #    pose = rotate_around_x(pose)
                transform = esper.component_for_entity(entity_id, Transform)
                transform.set_X_PT(pose)
        else:
            print("Optimization failed")


FACTOR_GRAPH_SOLVER = FactorGraphSolver()
