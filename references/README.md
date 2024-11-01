<!-- https://medium.com/@tarekeesa7/latest-yolov8-yolov9-guide-for-hyperparameter-tuning-and-data-augmentation-2024-469c69f295e0 -->

# Data Augmentation

When training a YOLO model, Ultralytics automatically applies various data augmentation techniques to improve model generalization and robustness. Here are the available data augmentation settings:

- **hsv_h=0.015**: Adjusts the hue to help the model generalize under different lighting and environmental conditions.
- **hsv_s=0.7**: Modifies the image's saturation.
- **hsv_v=0.4**: Alters the brightness of the image.
- **degrees=0.0**: Rotates the image randomly within a specific range.
- **translate=0.1**: Translates the image horizontally or vertically to improve detection of partially visible objects.
- **scale=0.5**: Scales the image, which aids in detecting objects at varying distances.
- **shear=0.0**: Shears the image, shifting parallel lines in the image to help detect objects from different angles.
- **perspective=0.0**: Applies a random perspective to the image for better scene understanding.
- **flipud=0.0**: Flips the image upside down.
- **fliplr=0.0**: Flips the image left-to-right.
- **bgr=0.0**: Guards against incorrect channel ordering. Avoid using if your dataset is correctly ordered.
- **mosaic=1.0**: Combines four images into a 2x2 image, enhancing the model's scene understanding and performance.
- **mixup=0.0**: Creates a composite image by merging two images and their labels.
- **copy_paste=0.0**: Copies objects and pastes them into different scenes to increase variation.
- **erasing=0.4**: Randomly erases portions of the image, making the model better at predicting hidden or partially visible objects.
- **crop_fraction=1.0**: Crops a fraction of the image for varied object sizes and placements.

These techniques help improve the YOLO model's adaptability to diverse real-world scenarios.
