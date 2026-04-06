import os
import glob

# 清理中间模型文件，只保留最佳模型和最终模型
def cleanup_models(root_dir):
    """
    清理模型文件，只保留最佳模型
    :param root_dir: 根目录
    """
    # 遍历所有模型文件夹
    for model_dir in glob.glob(os.path.join(root_dir, "**", "models"), recursive=True):
        # 获取所有模型文件
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))
        
        if not model_files:
            continue
        
        # 分类文件
        best_models = []
        other_models = []
        
        for model_file in model_files:
            if "best_model" in model_file:
                best_models.append(model_file)
            else:
                other_models.append(model_file)
        
        # 保留最佳模型，删除其他模型
        if best_models:
            print(f"保留最佳模型: {len(best_models)}个")
            print(f"删除中间模型: {len(other_models)}个")
            
            for model_file in other_models:
                os.remove(model_file)
                print(f"删除: {model_file}")
        else:
            print(f"未找到最佳模型，保留所有模型: {model_dir}")

if __name__ == "__main__":
    # 清理results目录
    results_dir = "results"
    cleanup_models(results_dir)
    print("清理完成！")
