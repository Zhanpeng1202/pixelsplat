ffmpeg \
    -framerate 30 \
    -i /data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/outputs/rollerblade/test_for_4d/rb4d_2/color/%06d.png \
    -c:v libx264 \
    -pix_fmt yuv420p \
    /data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/video/rollerblade.mp4
