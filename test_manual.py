#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, 'src')

from detection.base_detector import BaseDetector, DetectionResult, detector_registry
from datetime import datetime, date

def test_basic_functionality():
    """Quick test of basic functionality"""
    
    # Test DetectionResult creation
    print("Testing DetectionResult creation...")
    result = DetectionResult(
        metric_name="test_metric",
        expected_value=100.0,
        actual_value=150.0,
        deviation_percentage=0.5,
        severity="high",
        detection_method="threshold",
        timestamp=datetime.now(),
        alert_sent=False
    )
    print(f"✓ DetectionResult created: {result.metric_name}")
    
    # Test DetectionResult properties
    assert result.is_anomaly == True, "High deviation should be anomaly"
    print("✓ is_anomaly property works")
    
    # Test DetectionResult validation
    try:
        invalid_result = DetectionResult(
            metric_name="test",
            expected_value=100.0,
            actual_value=150.0,
            deviation_percentage=0.5,
            severity="invalid",
            detection_method="threshold",
            timestamp=datetime.now(),
            alert_sent=False
        )
        assert False, "Should have raised ValueError for invalid severity"
    except ValueError:
        print("✓ Severity validation works")
    
    # Test detector registry
    print("Testing detector registry...")
    detector_registry.clear()
    
    @detector_registry.register("test_detector")
    class TestDetector(BaseDetector):
        def detect(self, start_date: date, end_date: date) -> list:
            return [result]
        
        def validate_config(self) -> bool:
            return "metric" in self.config
    
    print("✓ Detector registration works")
    
    # Test detector creation
    config = {"metric": "views", "threshold": 0.2}
    detector = detector_registry.create_detector("test_detector", config)
    print(f"✓ Detector created: {detector.name}")
    
    # Test detector validation
    assert detector.validate_config() == True, "Config should be valid"
    print("✓ Config validation works")
    
    # Test detection
    results = detector.detect(date.today(), date.today())
    assert len(results) == 1, "Should return one result"
    print("✓ Detection method works")
    
    print("\n🎉 All basic functionality tests passed!")

if __name__ == "__main__":
    test_basic_functionality()