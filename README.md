# VJTAssignment

Edge cases for the images:
1. Skipped images with missing segmentation masks
2. Overlapping masks for same images get replaced by the newer ones
3. Corrupted segmentation masks also get skipped
4. Skip images with category id other than the original 80 ids
