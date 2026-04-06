import os
import os.path

# 检查视频文件
def check_video_files():
    video_dir = 'results/training_videos'
    print(f"检查视频目录: {video_dir}")
    print("=" * 50)
    
    if not os.path.exists(video_dir):
        print("错误: 视频目录不存在")
        return
    
    files = os.listdir(video_dir)
    print(f"目录中有 {len(files)} 个文件:")
    print("-" * 50)
    
    for file in files:
        file_path = os.path.join(video_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print(f"{file:40} {size_mb:.2f} MB")
        else:
            print(f"{file:40} [目录]")
    
    print("=" * 50)
    print("检查完成!")

if __name__ == "__main__":
    check_video_files()
