
from waynon.components.simple import Detector
from waynon.detectors.measurement_processor import MeasurementProcessor

class CharucoBoardDetector(Detector):

    def get_processor(self) -> MeasurementProcessor:
        from waynon.detectors.charuco_processor import CHARUCO_PROCESSOR
        return CHARUCO_PROCESSOR

    def draw_property(self, nursery, entity_id):
        super().draw_property(nursery, entity_id)
    
    def property_order(self):
        return 200 