# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from imgui_bundle import imgui


from .camera import PinholeCamera
from .component import Component
from .robot import Robot
from .tree_utils import *


class Measurement(Component):

    def on_selected(self, nursery, entity_id, just_selected):
        from .image_measurement import ImageMeasurement, activate_image_measurement
        from .joint_measurement import JointMeasurement, activate_joint_measurement
        
        if just_selected:
            for i, image_measurement_id in enumerate(find_children_with_component(entity_id, ImageMeasurement)):
                activate_image_measurement(image_measurement_id)
                if i == 0:
                    esper.dispatch_event("image_viewer", image_measurement_id)

            for joint_measurement_id in find_children_with_component(entity_id, JointMeasurement):
                activate_joint_measurement(joint_measurement_id)
           