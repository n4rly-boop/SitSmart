#!/bin/bash

# This script sends each image in parsing/total_images to the /api/features endpoint
# and saves the extracted features into dataset/features.csv with one column per feature:
# image_name, shoulder_line_angle_deg, head_tilt_deg, head_to_shoulder_distance_px, head_to_shoulder_distance_ratio, shoulder_width_px

IMAGES_DIR="parsing/total_images"
OUTPUT_CSV="dataset/features.csv"
API_URL="http://localhost:8000/api/features"

mkdir -p "$(dirname "$OUTPUT_CSV")"

# Write CSV header (one column per feature)
echo "image_name,shoulder_line_angle_deg,head_tilt_deg,head_to_shoulder_distance_px,head_to_shoulder_distance_ratio,shoulder_width_px" > "$OUTPUT_CSV"

# Loop through all image files in the directory
for img_path in "$IMAGES_DIR"/*; do
    if [[ -f "$img_path" ]]; then
        img_name="$IMAGES_DIR/$(basename "$img_path")"
        # Send POST request with the image file
        response=$(curl -s -X POST "$API_URL" \
            -H "accept: application/json" \
            -H "Content-Type: multipart/form-data" \
            -F "image=@$img_path")
        # Parse feature values and write a wide CSV row
        csv_values=$(echo "$response" | python3 -c "import sys, json; r=json.load(sys.stdin); f=r.get('features') or {}; keys=['shoulder_line_angle_deg','head_tilt_deg','head_to_shoulder_distance_px','head_to_shoulder_distance_ratio','shoulder_width_px']; vals=[('' if f.get(k) is None else str(f.get(k))) for k in keys]; print(','.join(vals))")
        echo "\"$img_name\",$csv_values" >> "$OUTPUT_CSV"
        echo "Processed $img_name"
    fi
done

echo "Feature extraction complete. Results saved to $OUTPUT_CSV"
