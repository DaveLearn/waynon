# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import esper
import numpy as np
import trio
from imgui_bundle import imgui
from PIL import Image

from .camera import PinholeCamera
from .component import Component
from .measurement import Measurement
from .node import Node
from .pose_group import PoseGroup
from .tree_utils import *

def activate_image_measurement(image_measurement_id):

    # Set the image on the camera
    image_measurement = esper.component_for_entity(
        image_measurement_id, ImageMeasurement
    )
    camera_entity_id = image_measurement.camera_id
    if esper.entity_exists(camera_entity_id):
        # set its texture
        camera = esper.try_component(camera_entity_id, PinholeCamera)
        if camera:
            camera.update_image(image_measurement.get_image_u())



class ImageMeasurement(Component):
    camera_id: int
    image_path: str

    def get_image_u(self):
        from .scene_utils import DATA_PATH
        image_path = DATA_PATH / self.image_path
        return np.array(Image.open(image_path))

    def property_order(self):
        return 100

    def get_camera(self):
        if not esper.entity_exists(self.camera_id):
            return None
        return esper.component_for_entity(self.camera_id, PinholeCamera)

    def draw_property(self, nursery, entity_id):
        imgui.separator()
        camera = self.get_camera()
        if camera is not None:
            node = get_node(self.camera_id)
            imgui.text(f"Camera: {node.name}")

        imgui.text(f"Image Path: {self.image_path}")

    def derive_name(self):
        if not esper.entity_exists(self.camera_id):
            return "Image"
        
        camera_node = get_node(self.camera_id)
        return f"Image - {camera_node.name}"

    def on_selected(self, nursery: trio.Nursery, entity_id: int, just_selected: bool):
        if just_selected:
            # Display the image
            esper.dispatch_event("image_viewer", entity_id)
            activate_image_measurement(entity_id)
    
    @staticmethod
    def default_name():
        return "Image"

    def _fix_on_load(self, new_to_old_entity_ids):
        self.camera_id = new_to_old_entity_ids.get(self.camera_id, self.camera_id)
