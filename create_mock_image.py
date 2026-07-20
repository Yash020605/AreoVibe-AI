import numpy as np
import cv2
import os

# Create a dummy "healthy crop" image (mostly green with some varied noise)
img = np.zeros((400, 400, 3), dtype=np.uint8)
img[:, :] = [34, 139, 34] # Forest Green (BGR: 34, 139, 34 -> actually BGR is 34, 139, 34? No, BGR for forest green is 34, 139, 34 if it was RGB.
# Wait, let's just make it visually green in RGB. B: 34, G: 139, R: 34
cv2.randn(img, (34, 139, 34), (20, 30, 20))

# Add a "disease" spot (yellowish brown)
cv2.circle(img, (200, 200), 50, (60, 100, 150), -1)

cv2.imwrite("test.jpg", img)
print("Created mock drone capture: test.jpg")
