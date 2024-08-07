import os
import cv2
import base64
import requests
import json


# Ask user for the video folder path
def get_video_folder_path():
    folder_path = input("Enter the path to the folder containing the video files: ")
    if not os.path.exists(folder_path):
        print("The specified folder does not exist. Please try again.")
        return get_video_folder_path()
    return folder_path


# List all video files in the folder
def list_video_files(folder_path):
    video_files = [file for file in os.listdir(folder_path) if file.endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"))]
    if not video_files:
        print("No video files found in the specified folder.")
        return list_video_files(get_video_folder_path())
    return video_files


# Ask user how many episodes are in this season
def get_number_of_episodes():
    num_episodes = input("Enter the number of episodes in this season: ")
    if not num_episodes.isdigit():
        print("Please enter a valid number.")
        return get_number_of_episodes()
    return int(num_episodes)


# Compare the number of video files to the number of episodes and calculate the number of episodes per video
def calculate_episodes_per_video(video_files, num_episodes):
    num_videos = len(video_files)
    # Calculate the number of episodes per video file
    episodes_per_video = num_episodes // num_videos
    # Calculate the remaining episodes after distributing them evenly
    remaining_episodes = num_episodes % num_videos
    # Create a list to hold the number of episodes in each file
    episodes_per_file = [episodes_per_video] * num_videos
    # Distribute the remaining episodes evenly among the video files
    for i in range(remaining_episodes):
        episodes_per_file[i] += 1
    return episodes_per_file


# Get the video length in seconds
def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return duration


# Calculate the duration of each episode in a video file
def calculate_episode_duration(video_path, episodes_per_file):
    check_points = []
    video_length = get_video_length(video_path)
    for episodes in episodes_per_file:
        if episodes > 1:
            interval = video_length / episodes
            for i in range(1, episodes):
                check_points.append(interval * i)

    return sorted(set(check_points))


# Preprocess a frame before saving it
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Use morphological operations to enhance the text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded


def extract_frames(video_path, check_points, output_folder="frames/"):
    # Get the directory of the video file
    video_dir = os.path.dirname(video_path)
    # Define the folder where frames will be saved
    frame_output_folder = os.path.join(video_dir, output_folder)

    # Check if the output folder exists and create it if not
    if not os.path.exists(frame_output_folder):
        os.makedirs(frame_output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_paths = []

    for time in check_points:
        time_marks_to_check = [time + i for i in range(-30, 31, 1)]

        for time_mark in time_marks_to_check:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_mark * 1000)
            success, frame = cap.read()
            if success:
                frame_time = f"{time_mark:.0f}"
                frame_path = os.path.join(frame_output_folder, f"{os.path.basename(video_path)}_{frame_time}.png")
                processed_frame = preprocess_frame(frame)
                cv2.imwrite(frame_path, processed_frame)
                frame_paths.append(frame_path)

    # Release the video capture object
    cap.release()
    return frame_paths


# Convert an image to base64 encoding
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return b64_string


# Detect text in an image using an API
def detect_text_with_api(image_path):
    api_url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    b64_image = image_to_base64(image_path)
    data = {
        "model": "llava-llama3",
        "prompt": "What text is in this picture? If there isn't any, reply with 'None'",
        "stream": False,
        "images": [b64_image]
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    result_text = response.json().get("response", "")
    print(f"Detected text in {image_path}:\n{result_text}\n{'-' * 80}\n")  # Debugging output
    return result_text


# Find the end timestamps of a video by detecting the text "the end" from the provided frame paths
def find_the_end_timestamps(frame_paths):
    text_results = []
    end_times = []
    end_detected = False
    for frame_path in frame_paths:
        text = detect_text_with_api(frame_path)
        text_results.append((frame_path, text))
        if "the end" in text.lower():
            end_detected = True
        if end_detected and "none" in text.lower():
            end_times.append(float(os.path.basename(frame_path).split("_")[-1].replace(".png", "")))
            end_detected = False

    return end_times, text_results


# main function
def main():
    # Get the folder path containing the video files
    folder_path = get_video_folder_path()
    # List all video files in the folder
    video_files = list_video_files(folder_path)
    # Ask the user for the number of episodes in this season
    num_episodes = get_number_of_episodes()
    # Calculate the number of episodes per video file
    episodes_per_file = calculate_episodes_per_video(video_files, num_episodes)
    print(f"Episodes per file: {episodes_per_file}")
    # Calculate episode durations for each video file
    text_results_file = folder_path + "text_results.txt"
    for video_path in video_files:
        video_file_path = os.path.join(folder_path, video_path)
        print(f"Processing video: {video_path}")
        episode_durations = calculate_episode_duration(video_file_path, episodes_per_file)
        print(f"Episode durations for {video_path}: {episode_durations}")
        frame_paths = extract_frames(video_file_path, episode_durations)
        end_times, text_results = find_the_end_timestamps(frame_paths)
        print(f"End times for {video_path}: {end_times}")
        # Add text results to file on a new line
        with open(text_results_file, 'a') as f:
            f.write(f"Text results for {video_path}:\n")
            for frame_path, text in text_results:
                f.write(f"Frame: {frame_path}\n{text}\n{'-' * 80}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
