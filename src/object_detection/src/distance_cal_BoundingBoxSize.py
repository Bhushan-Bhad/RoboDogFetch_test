import math

class DistanceEstimator:
    def __init__(self, focal_length_px, real_person_height_m=1.7):
        self.focal_length_px = focal_length_px  # Derived from camera parameters
        self.real_person_height_m = real_person_height_m

    def estimate_distance(self, person_height_px):
        # Estimate distance based on person height in the image
        distance = (self.real_person_height_m * self.focal_length_px) / person_height_px
        return distance

# Example usage:
focal_length_px = 600  # You can calculate this from camera's FoV
estimator = DistanceEstimator(focal_length_px)
person_height_in_image = 150  # Example bounding box height from object detection
distance_to_person = estimator.estimate_distance(person_height_in_image)

print(f"Estimated distance to person: {distance_to_person} meters")
