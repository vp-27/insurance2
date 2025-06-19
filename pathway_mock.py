"""
Mock implementation of Pathway for development purposes
"""

class MockPathway:
    def __init__(self):
        pass
    
    def run(self):
        print("Mock Pathway pipeline running...")
        return "Mock pipeline completed"

# Mock the pathway module
import sys
sys.modules['pathway'] = type(sys)('pathway')
sys.modules['pathway'].pw = MockPathway()
