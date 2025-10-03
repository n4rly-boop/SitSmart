#!/bin/bash

# This script sends each image in parsing/total_images/{good,bad} to the /api/features endpoint
# and saves the extracted features into dataset/features.csv with one column per feature:
# image_name,shoulder_line_angle_deg,head_tilt_deg,head_to_shoulder_distance_px,
# head_to_shoulder_distance_ratio,shoulder_width_px,label

IMAGES_DIR="parsing/total_images"
OUTPUT_CSV="dataset/features.csv"
API_URL="http://localhost:8000/api/features"

mkdir -p "$(dirname "$OUTPUT_CSV")"

# Write CSV header (add label column)
echo "image_name,shoulder_line_angle_deg,head_tilt_deg,head_to_shoulder_distance_px,head_to_shoulder_distance_ratio,shoulder_width_px,label" > "$OUTPUT_CSV"

# Loop through all images in subfolders (good / bad)
for label in good bad; do
    for img_path in "$IMAGES_DIR/$label"/*; do
        if [[ -f "$img_path" ]]; then
            img_name="$(basename "$img_path")"
            # Send POST request with the image file
            response=$(curl -s -X POST "$API_URL" \
                -H "accept: application/json" \
                -H "Content-Type: multipart/form-data" \
                -F "image=@$img_path")
            # Parse feature values and write a wide CSV row
            csv_values=$(echo "$response" | python3 -c "
import sys, json
r=json.load(sys.stdin)
f=r.get('features') or {}
keys=['shoulder_line_angle_deg','head_tilt_deg','head_to_shoulder_distance_px','head_to_shoulder_distance_ratio','shoulder_width_px']
vals=[('' if f.get(k) is None else str(f.get(k))) for k in keys]
print(','.join(vals))")
            echo "\"$img_name\",$csv_values,$label" >> "$OUTPUT_CSV"
            echo "Processed $img_name [$label]"
        fi
    done
done

echo "Feature extraction complete. Results saved to $OUTPUT_CSV"
