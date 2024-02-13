IMG_DIR=/ubc/cs/research/kmyi/matthew/projects/evimo_formatter/debug/colcam_set/rgb/1x

# ffmpeg -framerate 16 -i $IMG_DIR/%05d.png -c:v libx264 -pix_fmt yuv420p vid.mp4
ffmpeg -framerate 16 -i $IMG_DIR/%05d.png -vf "drawtext=fontfile=/path/to/font.ttf: text='%{frame_num}': start_number=0: x=10: y=10: fontcolor=white: fontsize=24: box=1: boxcolor=black@0.5: boxborderw=5" -c:v libx264 -pix_fmt yuv420p vid.mp4
