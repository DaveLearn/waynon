
import cv2.aruco as aruco

from imgui_bundle import imgui

from waynon.components.simple import Detector
from waynon.detectors.measurement_processor import MeasurementProcessor
from waynon.components.aruco_marker import aruco_dict_names, aruco_dict_values

class CharucoBoardDetector(Detector):
    marker_dict: int = aruco.DICT_4X4_1000

    def get_processor(self) -> MeasurementProcessor:
        from waynon.detectors.charuco_processor import CHARUCO_PROCESSOR
        return CHARUCO_PROCESSOR

    def draw_property(self, nursery, entity_id):
        super().draw_property(nursery, entity_id)
        imgui.separator()
        
        current_item = aruco_dict_values.index(self.marker_dict)
        if imgui.begin_combo("Dict", aruco_dict_names[current_item]):
            for i, (name, value) in enumerate(zip(aruco_dict_names, aruco_dict_values)):
                selected = current_item == i
                res, _ = imgui.selectable(name, selected)
                if res:
                    print(f"Selected {name}, {value}")
                    self.marker_dict = value
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
    
    def property_order(self):
        return 200 