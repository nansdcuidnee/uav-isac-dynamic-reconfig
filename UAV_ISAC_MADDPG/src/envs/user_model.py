import numpy as np

class UserModel:
    """
    用户模型
    - 用户分布生成
    - 用户位置管理
    """

    def __init__(self, area_size=200.0):
        """
        初始化用户模型
        :param area_size: 区域大小
        """
        self.area_size = area_size

    def generate_users(self, num_users):
        """
        生成用户位置
        :param num_users: 用户数量
        :return: 用户位置
        """
        # 生成随机用户位置
        user_pos = np.random.uniform(0, self.area_size, (num_users, 3))
        user_pos[:, 2] = 0.0  # 地面高度
        return user_pos

    def generate_clustered_users(self, num_users, num_clusters=3):
        """
        生成聚类用户位置
        :param num_users: 用户数量
        :param num_clusters: 聚类数量
        :return: 用户位置
        """
        user_pos = []
        
        # 生成聚类中心
        cluster_centers = np.random.uniform(0, self.area_size, (num_clusters, 2))
        
        # 为每个聚类生成用户
        users_per_cluster = num_users // num_clusters
        for i in range(num_clusters):
            cluster_users = np.random.normal(
                loc=cluster_centers[i],
                scale=10.0,
                size=(users_per_cluster, 2)
            )
            # 确保用户在区域内
            cluster_users = np.clip(cluster_users, 0, self.area_size)
            # 添加高度信息
            cluster_users_3d = np.column_stack([cluster_users, np.zeros(users_per_cluster)])
            user_pos.append(cluster_users_3d)
        
        # 处理剩余用户
        remaining_users = num_users % num_clusters
        if remaining_users > 0:
            remaining_users_pos = np.random.uniform(0, self.area_size, (remaining_users, 3))
            remaining_users_pos[:, 2] = 0.0
            user_pos.append(remaining_users_pos)
        
        return np.vstack(user_pos)

    def get_user_distribution(self, user_pos):
        """
        获取用户分布统计信息
        :param user_pos: 用户位置
        :return: 分布统计信息
        """
        x_coords = user_pos[:, 0]
        y_coords = user_pos[:, 1]
        
        stats = {
            'mean_x': np.mean(x_coords),
            'mean_y': np.mean(y_coords),
            'std_x': np.std(x_coords),
            'std_y': np.std(y_coords),
            'min_x': np.min(x_coords),
            'max_x': np.max(x_coords),
            'min_y': np.min(y_coords),
            'max_y': np.max(y_coords)
        }
        
        return stats