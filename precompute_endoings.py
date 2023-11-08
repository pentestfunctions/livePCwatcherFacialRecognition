import face_recognition
import argparse
import json
import os
from multiprocessing import Pool, cpu_count
import glob
from PIL import Image

# Function to get all image files from a directory and its subdirectories
def get_image_files(folder, existing_files):
    image_types = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp')  # Add or remove image types if needed
    files_grabbed = []
    for files in image_types:
        path_pattern = os.path.join(folder, '**', files)
        files_grabbed.extend(glob.glob(path_pattern, recursive=True))
    # Filter out files that have already been processed
    return [file for file in files_grabbed if file not in existing_files]

def process_image(image_file):
    try:
        print(f"Processing {image_file}...")
        encodings, name = encode_faces(image_file)
        if encodings:
            encodings_list = [encoding.tolist() for encoding in encodings]
            return image_file, {"encodings": encodings_list, "name": name}
        else:
            #print(f"No faces found in {image_file}.")
            return image_file, None
    except Exception as e:
        print(f"Skipping {image_file} due to error: {e}")
        return image_file, None

# Function to encode faces in an image and get the name
def encode_faces(image_path, default_name='unknown'):
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        # Extract person's name from the image path or use default name
        name = os.path.basename(os.path.dirname(image_path)) or default_name
        return face_encodings, name
    except Image.UnidentifiedImageError:
        #print(f"Cannot identify image file {image_path}. It may be corrupted or in an unsupported format.")
        return [], default_name
    except Exception as e:
        #print(f"Error processing image {image_path}: {e}")
        return [], default_name

# Function to precompute face encodings and save them to a JSON file
def precompute_face_encodings(folder, save_file):
    # Load existing data if available
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            face_data = json.load(f)
    else:
        face_data = {}
    
    existing_files = set(face_data.keys())
    image_files = get_image_files(folder, existing_files)
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, image_files)

    # Update the face_data with new encodings
    for image_file, data in results:
        if data:
            face_data[image_file] = data

    with open(save_file, 'w') as f:
        json.dump(face_data, f, indent=4)

    print(f"Data saved to {save_file}")

def main():
    # Set default arguments for the script
    folder = 'faces'  # Default folder set to "faces"
    output = 'precomputed_encodings.json'  # Default output file name

    # You can still override the output file name with an argument, if needed
    parser = argparse.ArgumentParser(description='Precompute face encodings and save to a JSON file.')
    parser.add_argument('--output', help='Output JSON file name for the face encodings.', default=output)
    args = parser.parse_args()

    # Run precompute function with the default "faces" folder and the specified output file name
    precompute_face_encodings(folder, args.output)

if __name__ == "__main__":
    main()
