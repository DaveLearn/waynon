# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from PIL import Image
import numpy as np

import esper
import trio

from imgui_bundle import imgui

from .tree_utils import *
from .component import Component
from .node import Node
from .pose_group import PoseGroup
from .camera import PinholeCamera
from .measurement import Measurement


def activate_joint_measurement(joint_measurement_id):
    from .robot import Robot
    joint_measurement = esper.component_for_entity(joint_measurement_id, JointMeasurement)
                
    robot = esper.try_component(joint_measurement.robot_id, Robot)
    if robot:
        manager = robot.get_manager()
        if manager:
            manager.set_offline_q(np.asarray(joint_measurement.joint_values))


class JointMeasurement(Component):
    robot_id: int
    joint_values: list[float]

    def property_order(self):
        return 100

    def draw_property(self, nursery, entity_id):
        imgui.separator()
        for i, j in enumerate(self.joint_values):
            imgui.text(f"Joint {i}: {j}")
    
    
    def derive_name(self):
        if not esper.entity_exists(self.robot_id):
            return "Joints"
        
        robot_node = get_node(self.robot_id)
        return f"Joints - {robot_node.name}"
    

    def on_selected(self, nursery: trio.Nursery, entity_id: int, just_selected: bool):
        if just_selected:
            activate_joint_measurement(entity_id)


    @staticmethod
    def default_name():
        return "Joints"

    def _fix_on_load(self, new_to_old_entity_ids):
        self.robot_id = new_to_old_entity_ids[self.robot_id]

