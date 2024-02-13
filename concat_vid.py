from moviepy.editor import VideoFileClip, clips_array

def concatenate_videos_side_by_side(video_path1, video_path2, output_path):
    # Load the videos
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)

    # Find the max height to scale videos
    max_height = max(clip1.size[1], clip2.size[1])

    # Resize videos to the same height
    clip1_resized = clip1.resize(height=max_height)
    clip2_resized = clip2.resize(height=max_height)

    # Concatenate videos side by side
    final_clip = clips_array([[clip1_resized, clip2_resized]])

    # Write the result to a file
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

# Example usage
concatenate_videos_side_by_side('/ubc/cs/research/kmyi/matthew/projects/evimo_formatter/tmp/debug_colcam_set.mp4', 
                                '/ubc/cs/research/kmyi/matthew/projects/evimo_formatter/tmp/debug_trig_ecamset.mp4', 'output.mp4')
