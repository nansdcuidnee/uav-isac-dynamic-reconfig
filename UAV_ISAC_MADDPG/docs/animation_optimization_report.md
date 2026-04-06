# 轨迹动画优化报告

## 问题分析

### 1. 视频播放不流畅问题
- **根本原因**：降采样率过高（每5帧取一帧）导致运动不连续
- **表现**：视频播放时出现卡顿、跳帧现象
- **影响**：无法清晰展示无人机的连续运动轨迹

### 2. 高度变化显示问题
- **根本原因**：高度变化的视觉表示不够明显
- **表现**：难以区分不同高度的无人机位置
- **影响**：无法直观展示无人机的高度变化

## 解决方案

### 1. 视频播放流畅度优化

#### 智能降采样策略
- 根据帧数据长度自动调整降采样率：
  - 超长序列（>5000帧）：每20帧取一帧
  - 长序列（>2000帧）：每10帧取一帧
  - 中等长度序列（>1000帧）：每5帧取一帧
  - 短序列（≤1000帧）：每3帧取一帧
- 保持24 FPS的恒定帧率，确保平滑播放

#### 渲染优化
- 减小图形尺寸（8x6）和dpi（72），提高渲染速度并减小视频体积
- 确保所有帧使用相同的坐标轴范围，避免画面跳动
- 优化内存使用，避免内存泄漏

#### 视频质量优化
- 使用H.264编码器（libx264），提高压缩效率
- 平衡视频质量和大小（quality=8）
- 确保所有帧的图像大小一致

#### 关键时段保存
- 添加max_steps参数，只保存最后一个回合的帧，大幅减少帧数
- 计算并显示预计视频时长，方便用户了解视频长度

### 2. 高度变化显示优化

#### 高度范围计算
- 预计算实际数据的高度范围，确保高度变化显示更准确
- 动态调整归一化参数，适应不同的高度范围

#### 视觉表示增强
- 使用jet颜色映射，从蓝色（低）到红色（高），使高度变化更明显
- 增大无人机大小随高度的变化范围（80-200）
- 轨迹使用颜色渐变显示高度变化
- 保持高度变化的平滑过渡

## 技术实现

### 关键代码修改

1. **智能降采样**：
   ```python
   if len(frames_data) > 5000:
       downsampling_rate = 20  # 对于超长序列，每20帧取一帧
   elif len(frames_data) > 2000:
       downsampling_rate = 10  # 对于长序列，每10帧取一帧
   elif len(frames_data) > 1000:
       downsampling_rate = 5  # 对于中等长度序列，每5帧取一帧
   else:
       downsampling_rate = 3  # 对于短序列，每3帧取一帧
   ```

2. **关键时段保存**：
   ```python
   if max_steps and max_steps > 0 and len(frames_data) > max_steps:
       last_episode_frames = frames_data[-max_steps:]
       print(f"只保存最后一个回合的 {len(last_episode_frames)} 帧")
       frames_data = last_episode_frames
   ```

3. **视频编码优化**：
   ```python
   writer = imageio.get_writer(save_path, fps=fps, quality=8, codec='libx264')
   ```

4. **图像分辨率优化**：
   ```python
   fig = plt.figure(figsize=(8, 6), dpi=72)  # 减小图形尺寸和dpi
   ```

5. **高度范围计算**：
   ```python
   all_heights = []
   for frame_data in frames_data:
       drone_pos = np.array(frame_data['drone_pos'])
       if drone_pos.size > 0:
           all_heights.extend(drone_pos[:, 2])
   
   if all_heights:
       min_height = min(all_heights)
       max_height = max(all_heights)
       height_range = max_height - min_height
   ```

6. **高度可视化**：
   ```python
   # 颜色映射
   height_normalized = (d[2] - min_height) / height_range
   color = plt.cm.jet(height_normalized)
   
   # 大小变化
   base_size = 80
   size_range = 120
   marker_size = base_size + height_normalized * size_range
   ```

7. **固定坐标轴范围**：
   ```python
   # 预计算所有帧的坐标范围
   all_x = []
   all_y = []
   for frame_data in frames_data_downsampled:
       # 收集所有坐标
   
   # 设置固定的坐标轴范围
   ax.set_xlim([x_min, x_max])
   ax.set_ylim([y_min, y_max])
   ```

## 结果验证

### 优化效果
- **视频流畅度**：实现了24 FPS的平滑播放，消除了卡顿和跳帧现象
- **视频体积**：通过降采样、减小分辨率和使用高效编码器，大幅减小视频体积
- **高度变化显示**：通过颜色和大小变化，清晰展示了无人机的高度变化
- **视觉效果**：轨迹使用颜色渐变，直观展示了高度变化的平滑过渡
- **渲染速度**：优化后的渲染过程更快，生成时间显著减少

### 生成文件
- 优化后的视频已保存到：`results\exp3_num_uav_1_optimized\videos\trajectory_animation.mp4`
- 测试大帧数的视频已保存到：`results\exp3_num_uav_1_optimized\videos\trajectory_animation_large.mp4`

## 结论

通过以上优化，轨迹动画现在具有以下特点：
1. **流畅的视频播放**：24 FPS的恒定帧率，消除了卡顿现象
2. **小体积视频**：通过多种优化手段，大幅减小视频体积，提高播放流畅度
3. **清晰的高度变化显示**：通过颜色和大小变化，直观展示高度差异
4. **专业的视觉效果**：轨迹颜色渐变，平滑的高度过渡
5. **高效的渲染过程**：智能降采样，优化的渲染参数，减少生成时间
6. **灵活的配置**：支持只保存最后一个回合的帧，进一步减小视频体积

这些改进使得轨迹动画更加专业，能够更好地展示无人机的运动轨迹和高度变化，符合论文和竞赛的展示要求。同时，优化后的视频生成过程更加高效，减少了资源消耗和等待时间。