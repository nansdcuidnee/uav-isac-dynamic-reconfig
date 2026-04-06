import os
import csv
import json
import numpy as np

class DataExporter:
    """
    数据导出器，用于将仿真数据导出为Unity可读取的CSV和JSON格式
    """
    
    def __init__(self, export_dir='unity_data'):
        """
        初始化数据导出器
        :param export_dir: 导出目录
        """
        self.export_dir = export_dir
        # 创建导出目录
        os.makedirs(self.export_dir, exist_ok=True)
        
        # 创建StreamingAssets目录（Unity标准目录结构）
        self.streaming_assets_dir = os.path.join(self.export_dir, 'StreamingAssets')
        os.makedirs(self.streaming_assets_dir, exist_ok=True)
        
    def export_uav_trajectories(self, trajectory_data):
        """
        导出无人机轨迹数据
        :param trajectory_data: 轨迹数据列表
        """
        # 按无人机ID分组
        uav_data = {}
        for entry in trajectory_data:
            uav_id = entry['uav_id']
            if uav_id not in uav_data:
                uav_data[uav_id] = []
            uav_data[uav_id].append(entry)
        
        # 为每个无人机创建CSV文件
        for uav_id, data in uav_data.items():
            csv_path = os.path.join(self.export_dir, f'uav_{uav_id}.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # 写入表头
                writer.writerow(['time', 'x', 'y', 'z'])
                # 写入数据
                for entry in data:
                    time = entry['step']
                    x = entry['x']
                    y = entry['y']
                    z = entry['z']
                    writer.writerow([time, x, y, z])
            print(f"无人机 {uav_id} 轨迹数据已导出到 {csv_path}")
    
    def export_tasks(self, user_positions):
        """
        导出任务点数据
        :param user_positions: 用户位置数据
        """
        csv_path = os.path.join(self.export_dir, 'tasks.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['task_id', 'x', 'y', 'z', 'status'])
            # 写入数据
            for task_id, pos in enumerate(user_positions):
                x = pos[0]
                y = pos[1]
                z = pos[2]
                status = 'pending'
                writer.writerow([task_id + 1, x, y, z, status])
        print(f"任务点数据已导出到 {csv_path}")
    
    def export_communication(self, trajectory_data, num_uavs):
        """
        导出通信数据
        :param trajectory_data: 轨迹数据列表
        :param num_uavs: 无人机数量
        """
        csv_path = os.path.join(self.export_dir, 'communication.csv')
        
        # 按步骤分组
        step_data = {}
        for entry in trajectory_data:
            step = entry['step']
            if step not in step_data:
                step_data[step] = {}
            step_data[step][entry['uav_id']] = (entry['x'], entry['y'], entry['z'])
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['time', 'uav1', 'uav2', 'distance', 'signal'])
            
            # 计算每一步的通信数据
            for step, positions in step_data.items():
                # 计算所有无人机对之间的通信
                for uav1 in range(num_uavs):
                    for uav2 in range(uav1 + 1, num_uavs):
                        if uav1 in positions and uav2 in positions:
                            pos1 = positions[uav1]
                            pos2 = positions[uav2]
                            # 计算距离
                            distance = np.sqrt(
                                (pos1[0] - pos2[0])**2 +
                                (pos1[1] - pos2[1])**2 +
                                (pos1[2] - pos2[2])**2
                            )
                            # 计算信号强度（简单模型：距离越近信号越强）
                            signal = max(0, 1 - distance / 200)  # 200为最大通信距离
                            writer.writerow([step, uav1, uav2, distance, signal])
        print(f"通信数据已导出到 {csv_path}")
    
    def export_energy(self, energy_data):
        """
        导出能量数据
        :param energy_data: 能量数据列表
        """
        csv_path = os.path.join(self.export_dir, 'energy.csv')
        
        # 按无人机ID分组
        uav_energy = {}
        for entry in energy_data:
            uav_id = entry['uav_id']
            if uav_id not in uav_energy:
                uav_energy[uav_id] = []
            uav_energy[uav_id].append(entry)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['time', 'uav_id', 'energy'])
            
            # 写入数据
            for uav_id, data in uav_energy.items():
                for entry in data:
                    time = entry['step']
                    energy = entry['energy']
                    writer.writerow([time, uav_id, energy])
        print(f"能量数据已导出到 {csv_path}")
    
    def export_uav_trajectory(self, trajectory_data):
        """
        导出无人机轨迹数据为单个CSV文件
        :param trajectory_data: 轨迹数据列表
        """
        csv_path = os.path.join(self.export_dir, 'uav_trajectory.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['time', 'uav_id', 'x', 'y', 'z'])
            
            # 写入数据
            for entry in trajectory_data:
                time = entry['step']
                uav_id = entry['uav_id']
                x = entry['x']
                y = entry['y']
                z = entry['z']
                writer.writerow([time, uav_id, x, y, z])
        print(f"无人机轨迹数据已导出到 {csv_path}")
    
    def export_json_trajectory(self, trajectory_data):
        """
        导出轨迹数据为JSON格式
        :param trajectory_data: 轨迹数据列表
        """
        json_path = os.path.join(self.streaming_assets_dir, 'trajectory_data.json')
        
        # 按无人机ID组织数据
        organized_data = []
        for entry in trajectory_data:
            organized_data.append({
                'episode': entry['episode'],
                'step': entry['step'],
                'uav_id': entry['uav_id'],
                'x': entry['x'],
                'y': entry['y'],
                'z': entry['z']
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(organized_data, f, indent=2, ensure_ascii=False)
        print(f"轨迹数据（JSON）已导出到 {json_path}")
    
    def export_json_communication(self, trajectory_data, num_uavs):
        """
        导出通信数据为JSON格式
        :param trajectory_data: 轨迹数据列表
        :param num_uavs: 无人机数量
        """
        json_path = os.path.join(self.streaming_assets_dir, 'communication_data.json')
        
        # 按步骤分组
        step_data = {}
        for entry in trajectory_data:
            step = entry['step']
            if step not in step_data:
                step_data[step] = {}
            step_data[step][entry['uav_id']] = (entry['x'], entry['y'], entry['z'])
        
        # 生成通信数据
        communication_data = []
        for step, positions in step_data.items():
            # 计算所有无人机对之间的通信
            for uav1 in range(num_uavs):
                for uav2 in range(uav1 + 1, num_uavs):
                    if uav1 in positions and uav2 in positions:
                        pos1 = positions[uav1]
                        pos2 = positions[uav2]
                        # 计算距离
                        distance = np.sqrt(
                            (pos1[0] - pos2[0])**2 +
                            (pos1[1] - pos2[1])**2 +
                            (pos1[2] - pos2[2])**2
                        )
                        # 计算信号强度
                        signal_strength = max(0, 1 - distance / 200)
                        communication_data.append({
                            'time': step,
                            'uav1': uav1,
                            'uav2': uav2,
                            'distance': distance,
                            'signal_strength': signal_strength
                        })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(communication_data, f, indent=2, ensure_ascii=False)
        print(f"通信数据（JSON）已导出到 {json_path}")
    
    def export_json_tasks(self, user_positions):
        """
        导出任务数据为JSON格式
        :param user_positions: 用户位置数据
        """
        json_path = os.path.join(self.streaming_assets_dir, 'tasks.json')
        
        tasks_data = []
        for task_id, pos in enumerate(user_positions):
            tasks_data.append({
                'task_id': task_id + 1,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'status': 'pending'
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=2, ensure_ascii=False)
        print(f"任务数据（JSON）已导出到 {json_path}")
    
    def export_json_user_positions(self, user_positions):
        """
        导出用户位置数据为JSON格式
        :param user_positions: 用户位置数据
        """
        json_path = os.path.join(self.streaming_assets_dir, 'user_positions.json')
        
        user_data = []
        for user_id, pos in enumerate(user_positions):
            user_data.append({
                'user_id': user_id,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2]
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)
        print(f"用户位置数据（JSON）已导出到 {json_path}")

def export_simulation_data(environment, trajectory_data, energy_data=None, export_dir='unity_data'):
    """
    导出仿真数据
    :param environment: 环境对象
    :param trajectory_data: 轨迹数据
    :param energy_data: 能量数据
    :param export_dir: 导出目录
    """
    exporter = DataExporter(export_dir)
    
    # 导出无人机轨迹数据（CSV）
    exporter.export_uav_trajectories(trajectory_data)
    
    # 获取用户位置数据
    _, user_pos = environment.get_positions()
    
    # 导出任务点数据（CSV）
    exporter.export_tasks(user_pos)
    
    # 导出通信数据（CSV）
    num_uavs = environment.num_drones
    exporter.export_communication(trajectory_data, num_uavs)
    
    # 导出能量数据（CSV）
    if energy_data:
        exporter.export_energy(energy_data)
    else:
        # 如果没有能量数据，生成默认数据
        default_energy_data = []
        for entry in trajectory_data:
            default_energy_data.append({
                'step': entry['step'],
                'uav_id': entry['uav_id'],
                'energy': 100.0 - (entry['step'] * 0.1)  # 简单的能量消耗模型
            })
        exporter.export_energy(default_energy_data)
    
    # 导出无人机轨迹数据（单个CSV文件）
    exporter.export_uav_trajectory(trajectory_data)
    
    # 导出JSON格式数据（用于Unity）
    exporter.export_json_trajectory(trajectory_data)
    exporter.export_json_communication(trajectory_data, num_uavs)
    exporter.export_json_tasks(user_pos)
    exporter.export_json_user_positions(user_pos)
    
    print(f"所有数据已导出到 {export_dir} 目录")
    print("\n系统架构说明：")
    print("Python 仿真系统负责生成 JSON 数据文件，")
    print("Unity 仅读取这些数据进行三维可视化与仿真动画播放。")
    print("\nUnity 读取路径：")
    print(f"{export_dir}/StreamingAssets/")
    print("  - trajectory_data.json")
    print("  - user_positions.json")
    print("  - communication_data.json")
    print("  - tasks.json")
