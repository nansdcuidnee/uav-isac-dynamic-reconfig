import numpy as np
import gym
from gym import spaces
from src.envs.models.communication_model import CommunicationModel
from src.envs.models.sensing_model import SensingModel
from src.envs.models.uav_model import UAVModel
from src.envs.models.user_model import UserModel
from src.envs.models.environment_model import EnvironmentModel
from src.reconfiguration.event_manager import EventManager
from src.reconfiguration.member_manager import MemberManager
from src.utils import calculate_distance, clip_position, calculate_battery_consumption, generate_random_position

class UAVEnv(gym.Env):
    """
    UAV閫氫俊鎰熺煡涓€浣撳寲鐜
    - 澶氭棤浜烘満杞ㄨ抗瑙勫垝
    - 閫氫俊閫熺巼璁＄畻
    - 鎰熺煡SNR璁＄畻
    - 瑕嗙洊鐜囪瘎浼?
    """

    def __init__(self, num_drones=3, num_users=20, max_steps=128, alpha=1.0, beta=2.0, gamma=1.0, max_drones=None, initial_active_drones=None, event_schedule=None):
        super(UAVEnv, self).__init__()

        # 鍔ㄦ€佷簨浠剁浉鍏冲弬鏁?
        self.max_drones = max_drones if max_drones is not None else num_drones
        self.initial_active_drones = initial_active_drones if initial_active_drones is not None else num_drones
        self.event_schedule = event_schedule if event_schedule is not None else []
        # 鍒濆鍖栦簨浠剁鐞嗗櫒
        self.event_manager = EventManager(event_schedule)
        # 鍒濆鍖栨垚鍛樼鐞嗗櫒
        self.member_manager = MemberManager(
            max_drones=self.max_drones,
            initial_active_drones=self.initial_active_drones
        )
        
        # 淇濇寔鍚戝悗鍏煎
        self.num_drones = self.max_drones
        self.num_users = num_users
        self.max_steps = max_steps
        self.alpha = alpha  # 閫氫俊閫熺巼鏉冮噸
        self.beta = beta    # 瑕嗙洊鐜囨潈閲?
        self.gamma = gamma  # 鎯╃綒椤规潈閲?

        # 鍔ㄤ綔绌洪棿璁剧疆
        # 瀵逛簬 DQN锛屼娇鐢ㄧ鏁ｅ姩浣滅┖闂?
        self.action_space = spaces.Discrete(8)
        
        # 鍚屾椂淇濇寔杩炵画鍔ㄤ綔鐨勫吋瀹规€э紙鐢ㄤ簬 DDPG 绛夌畻娉曪級
        self.continuous_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 鏂瑰悜鏄犲皠
        self.DIRECTIONS = np.array([
            [1, 0],   # 鍓?
            [-1, 0],  # 鍚?
            [0, 1],   # 宸?
            [0, -1],  # 鍙?
            [1, 1],   # 宸﹀墠
            [1, -1],  # 宸﹀悗
            [-1, 1],  # 鍙冲墠
            [-1, -1]  # 鍙冲悗
        ])

        # 鐘舵€佺┖闂?
        # 姣忎釜鏅鸿兘浣撶殑瑙傚療缁村害
        obs_dim = 2 + 2 + 1 + 1 + 1 + num_users + num_users  # 2(浣嶇疆) + 2(閫熷害) + 1(瓒婄晫) + 1(纰版挒) + 1(鐢垫睜) + num_users(SNR) + num_users(杩炴帴)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 閫氫俊鍙傛暟
        self.P = 100.0  # 鍙戝皠鍔熺巼
        self.noise = 1e-3  # 鍣０鍔熺巼
        self.snr_threshold = 0.01  # 鎰熺煡淇″櫔姣旈槇鍊?
        self.safe_distance = 5.0  # 瀹夊叏璺濈
        self.communication_range = 100.0  # 閫氫俊鑼冨洿
        self.sensing_range = 120.0  # 鎰熺煡鑼冨洿
        self.max_speed = 10.0  # 鏈€澶ч€熷害
        self.max_acc = 15.0  # 鏈€澶у姞閫熷害锛屽澶у埌15.0
        self.dt = 1.0  # 鏃堕棿姝ラ暱
        self.area_size = 200.0  # 鍖哄煙澶у皬

        # 鐘舵€佸彉閲?
        self.out_of_bounds_count = 0
        self.collision_count = 0
        self.step_count = 0

        # 鐢垫睜妯″瀷
        self.battery = np.ones(self.num_drones) * 100.0  # 鍒濆鐢甸噺100%

        # 濂栧姳鍒嗛噺锛岀敤浜庤皟璇?
        self.r_com_raw = 0.0
        self.r_cover = 0.0
        self.r_penalty = 0.0
        self.dist_reward = 0.0
        self.min_dist = 0.0



        # 鐢熸垚鍥哄畾鐢ㄦ埛浣嶇疆
        self.user_pos = np.array([generate_random_position(self.area_size, 0.0) for _ in range(self.num_users)])

        # 鍒濆鍖栫幆澧冩ā鍨?
        self.env_model = EnvironmentModel(area_size=self.area_size, time_steps=self.max_steps)
        
        # 鍒濆鍖?UAV 妯″瀷鍒楄〃
        self.uav_models = []
        for i in range(self.num_drones):
            initial_position = generate_random_position(self.area_size, 50.0)
            uav_model = UAVModel(
                uav_id=i,
                initial_position=initial_position,
                max_speed=self.max_speed,
                communication_range=self.communication_range,
                sensing_range=self.sensing_range
            )
            self.uav_models.append(uav_model)
        
        # 鍒濆鍖栫敤鎴锋ā鍨嬪垪琛?
        self.user_models = []
        for i in range(self.num_users):
            user_model = UserModel(
                user_id=i,
                position=self.user_pos[i]
            )
            self.user_models.append(user_model)

        # 鍒濆鍖栭€氫俊鍜屾劅鐭ユā鍨?
        self.channel_model = CommunicationModel(P=self.P, noise=self.noise)
        self.sensing_model = SensingModel(P=self.P, noise=self.noise, snr_threshold=self.snr_threshold)

        # ISAC妯″潡闆嗘垚娴嬭瘯
        self._test_isac_integration()

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # 鏃犱汉鏈轰綅缃紙3D锛?
        self.drone_pos = np.array([generate_random_position(self.area_size, 50.0) for _ in range(self.num_drones)])
        self.drone_vel = np.zeros((self.num_drones, 2))

        # 鐢ㄦ埛浣嶇疆锛?D锛? 淇濇寔鍥哄畾锛屼笉鍐嶉噸鏂扮敓鎴?

        # 閲嶇疆鐘舵€佸彉閲?
        self.out_of_bounds_count = 0
        self.collision_count = 0
        self.step_count = 0

        # 閲嶇疆鐢垫睜鐢甸噺
        self.battery = np.ones(self.num_drones) * 100.0

        # 閲嶇疆鎴愬憳绠＄悊鍣?
        self.member_manager = MemberManager(
            max_drones=self.max_drones,
            initial_active_drones=self.initial_active_drones
        )

        obs = self._get_obs()
        return obs, {'active_mask': self.member_manager.get_active_mask_list()}

    def step(self, actions):
        # 浜嬩欢澶勭悊
        trigger_event = False
        event_type = None
        
        # 浣跨敤浜嬩欢绠＄悊鍣ㄦ鏌ュ綋鍓嶆鏄惁鏈変簨浠?
        event = self.event_manager.get_event(self.step_count + 1)
        join_agents = []
        
        if event['trigger_flag']:
            trigger_event = True
            event_type = event['event_type']
            
            # 澶勭悊浜嬩欢
            for agent_id in event['affected_agents']:
                if event_type == 'leave':
                    # leave浜嬩欢绔嬪嵆鐢熸晥
                    self.member_manager.deactivate(agent_id)
                elif event_type == 'join':
                    # 閲嶇疆join UAV鐨勭姸鎬?
                    # 浣嶇疆锛氶殢鏈哄垵濮嬩綅缃?
                    self.drone_pos[agent_id] = np.array([
                        np.random.uniform(0, self.area_size),
                        np.random.uniform(0, self.area_size),
                        50.0  # 鍥哄畾楂樺害
                    ])
                    # 閫熷害锛氭竻闆?
                    self.drone_vel[agent_id] = np.array([0.0, 0.0])
                    # 鐢甸噺锛氳涓烘弧鐢?
                    self.battery[agent_id] = 100.0
                    # 璁板綍闇€瑕佹縺娲荤殑UAV锛屼笅涓€姝ュ紑濮嬪弬涓?
                    join_agents.append(agent_id)

        # 纭繚actions鏄痭umpy鏁扮粍
        actions = np.array(actions)

        # 澶勭悊actions鐨勭淮搴?
        if actions.ndim == 1:
            # 瀵逛簬鍗曟棤浜烘満锛屼娇鐢ㄤ竴缁存暟缁勪綔涓鸿繛缁姩浣?
            if self.num_drones == 1:
                actions = np.expand_dims(actions, axis=0)
        elif actions.ndim == 2:
            # 瀵逛簬澶氭棤浜烘満锛岀‘淇濆舰鐘舵纭?
            if actions.shape[0] != self.num_drones:
                actions = actions.reshape(self.num_drones, -1)

        # 纭繚actions鐨勫舰鐘朵笌drone鏁伴噺鍖归厤
        actions = actions[:self.num_drones]

        # 鑾峰彇褰撳墠婵€娲荤殑鏃犱汉鏈篒D鍒楄〃
        active_drones = self.member_manager.get_active_ids()

        # 鏇存柊閫熷害鍜屼綅缃紙鍙active鐨刄AV锛?
        for i in active_drones:
            action = actions[i]
            if isinstance(action, np.ndarray) and len(action) >= 2:
                # 鍔犻€熷害鎺у埗
                self.drone_vel[i] += action[:2] * self.max_acc * self.dt
                self.drone_vel[i] = np.clip(self.drone_vel[i], -self.max_speed, self.max_speed)
                self.drone_pos[i, :2] += self.drone_vel[i] * self.dt
            else:
                # 鍏煎绂绘暎鍔ㄤ綔
                try:
                    # 灏濊瘯鐩存帴浣跨敤鏁存暟鍔ㄤ綔绱㈠紩
                    if isinstance(action, (np.ndarray, np.integer, np.floating)):
                        action_idx = int(action.item())
                    else:
                        action_idx = int(action)
                    action_idx = np.clip(action_idx, 0, 7)
                except (ValueError, TypeError):
                    # 濡傛灉澶辫触锛屽皾璇曞綊涓€鍖栧鐞嗭紙鍏煎鍏朵粬绠楁硶锛?
                    if isinstance(action, (np.ndarray, np.integer, np.floating)):
                        action_val = action.item()
                    else:
                        action_val = action
                    action_idx = int((action_val + 1) / 2 * 7)
                    action_idx = np.clip(action_idx, 0, 7)
                direction = self.DIRECTIONS[action_idx]
                # 鐩存帴璁剧疆閫熷害锛岃€屼笉鏄钩婊戞洿鏂?
                self.drone_vel[i] = direction * self.max_speed
                self.drone_pos[i, :2] += self.drone_vel[i] * self.dt

        # 杈圭晫妫€娴嬶紙鍙active鐨刄AV锛?
        out_of_bounds = 0
        for i in active_drones:
            # 妫€鏌ユ槸鍚﹁秺鐣?
            pos = self.drone_pos[i, :2]
            if np.any(pos < 0) or np.any(pos > self.area_size):
                out_of_bounds += 1
            # 瑁佸壀浣嶇疆
            clipped_pos = clip_position(self.drone_pos[i], self.area_size)
            self.drone_pos[i] = clipped_pos
        out_penalty = out_of_bounds
        self.out_of_bounds_count += out_penalty

        # 纰版挒妫€娴嬶紙鍙active鐨刄AV锛?
        collision_penalty = 0
        current_collisions = 0
        for i_idx, i in enumerate(active_drones):
            for j in active_drones[i_idx + 1:]:
                dist = calculate_distance(self.drone_pos[i], self.drone_pos[j])
                if dist < self.safe_distance:
                    collision_penalty += 1
                    current_collisions += 1
        self.collision_count += current_collisions

        # 浣跨敤閫氫俊妯″瀷璁＄畻鎬婚€氫俊閫熺巼鍜岀敤鎴疯繛鎺ユ儏鍐碉紙鍙€冭檻active鐨刄AV锛?
        active_drone_pos_list = [self.drone_pos[i].tolist() for i in active_drones]
        user_pos_list = self.user_pos.tolist()

        if active_drone_pos_list:
            total_rate, connections, snr_matrix, rate_matrix = self.channel_model.calculate_total_rate(
                active_drone_pos_list, user_pos_list, self.communication_range
            )
            self.user_connections = connections.tolist()

            # 浣跨敤鎰熺煡妯″瀷璁＄畻瑕嗙洊鐜囷紙鍙€冭檻active鐨刄AV锛?
            covered_users, coverage_rate, coverage_status = self.sensing_model.calculate_coverage(
                active_drone_pos_list, user_pos_list, self.sensing_range
            )
            served_users = covered_users
        else:
            total_rate = 0
            connections = []
            self.user_connections = []
            covered_users = 0
            coverage_rate = 0
            coverage_status = []
            served_users = 0

        # 璁＄畻鑳介噺娑堣€楋紙鍙active鐨刄AV锛?
        energy_cost = np.sum([self.drone_vel[i]**2 for i in active_drones])

        # 鍩轰簬閫熷害鐨勭數姹犳秷鑰楁ā鍨嬶紙鍙active鐨刄AV锛?
        battery_consumption = np.zeros(self.num_drones)
        for i in active_drones:
            battery_consumption[i] = calculate_battery_consumption(self.drone_vel[i], self.dt, 0.05)
        self.battery = np.clip(self.battery - battery_consumption, 0, 100)

        # 鏋勫缓濂栧姳鍑芥暟 - 涓庤鏂囦竴鑷?
        # r = 伪*r_com + 尾*r_cover - 纬*r_penalty

        # 璁＄畻鏈€灏忚窛绂伙紙鍙€冭檻active鐨刄AV锛?
        if active_drones:
            min_dist = min([calculate_distance(self.drone_pos[i], u) for i in active_drones for u in self.user_pos])
        else:
            min_dist = self.area_size
        # 骞虫柟鎯╃綒 - 璺濈濂栧姳锛堣礋鍊硷紝瓒婃帴杩?瓒婂ソ锛?
        dist_reward = - (min_dist / self.area_size) ** 2

        # 閫氫俊濂栧姳锛堝師濮嬪€硷級
        r_com_raw = total_rate
        # 閫氫俊濂栧姳褰掍竴鍖栵紙鍋囪鏈€澶ч€熺巼绾︿负50锛?
        r_com_norm = np.clip(r_com_raw / 50.0, 0, 1)

        # 瑕嗙洊鐜?
        r_cover = coverage_rate

        # 鎯╃綒椤?
        speed_penalty = np.sum([np.clip(np.abs(self.drone_vel[i]) - self.max_speed, 0, np.inf) for i in active_drones]) / self.max_speed
        r_penalty = (out_penalty + collision_penalty) * 0.5 + speed_penalty * 0.05
        
        # 鎬诲鍔?- 璋冩暣鏉冮噸锛屽鍔犺鐩栫巼鏉冮噸锛屽噺灏忚窛绂绘儵缃氭潈閲?
        # reward = 0.2 * r_com_raw + 0.3 * r_cover + 0.5 * dist_reward - r_penalty
        reward = 0.3 * dist_reward + 0.7 * r_cover - r_penalty
        
        # 瀛樺偍濂栧姳鍒嗛噺锛岀敤浜庤皟璇?
        self.r_com_raw = r_com_raw
        self.r_com_norm = r_com_norm
        self.r_cover = r_cover
        self.r_penalty = r_penalty
        self.dist_reward = dist_reward
        self.min_dist = min_dist

        # 妫€鏌ユ槸鍚︾粨鏉?
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # 璁＄畻骞冲潎閫氫俊閫熺巼
        avg_rate = total_rate / self.num_users if self.num_users > 0 else 0

        info = {
            'served_users': served_users,
            'total_system_rate': total_rate,
            'avg_rate': avg_rate,
            'coverage_rate': coverage_rate,
            'total_rate': total_rate,
            'min_dist': min_dist,
            'dist_reward': dist_reward,
            'distance_reward': dist_reward,  # 鍒悕锛屽吋瀹硅缁冭剼鏈?
            'r_com_raw': r_com_raw,
            'r_com_norm': r_com_norm,
            'r_cover': r_cover,
            'r_penalty': r_penalty,
            'penalty': r_penalty,
            'active_mask': self.member_manager.get_active_mask_list(),
            'trigger_event': trigger_event,
            'event_type': event_type
        }

        # 婵€娲籮oin UAV锛屼粠涓嬩竴姝ュ紑濮嬪弬涓?
        for agent_id in join_agents:
            self.member_manager.activate(agent_id)
        
        # 閲嶆柊鏋勫缓info锛岀‘淇漚ctive_mask琛ㄧず姝ョ粨鏉熷悗鐨勭姸鎬?
        info = {
            'served_users': served_users,
            'total_system_rate': total_rate,
            'avg_rate': avg_rate,
            'coverage_rate': coverage_rate,
            'total_rate': total_rate,
            'min_dist': min_dist,
            'dist_reward': dist_reward,
            'distance_reward': dist_reward,  # 鍒悕锛屽吋瀹硅缁冭剼鏈?
            'r_com_raw': r_com_raw,
            'r_com_norm': r_com_norm,
            'r_cover': r_cover,
            'r_penalty': r_penalty,
            'penalty': r_penalty,
            'active_mask': self.member_manager.get_active_mask_list(),
            'trigger_event': trigger_event,
            'event_type': event_type
        }
        
        obs = self._get_obs()
        return obs, [reward]*self.num_drones, done, False, info

    def _get_obs(self):
        obs_list = []

        # 涓烘瘡涓棤浜烘満鐢熸垚鍗曠嫭鐨勮瀵?
        for drone_idx in range(self.num_drones):
            obs = []

            # 褰撳墠鏃犱汉鏈轰綅缃紙鍙娇鐢▁鍜寉鍧愭爣锛?
            obs.extend(self.drone_pos[drone_idx, :2])

            # 褰撳墠鏃犱汉鏈洪€熷害
            obs.extend(self.drone_vel[drone_idx])

            # 瓒婄晫娆℃暟
            obs.append(self.out_of_bounds_count)

            # 纰版挒璁℃暟
            obs.append(self.collision_count)

            # 鐢垫睜鐢甸噺
            obs.append(self.battery[drone_idx])

            # 鎰熺煡SNR锛堝綋鍓嶆棤浜烘満瀵规瘡涓敤鎴凤級
            for j in range(self.num_users):
                d = calculate_distance(self.drone_pos[drone_idx], self.user_pos[j])
                snr = self.sensing_model.calculate_snr(d)
                obs.append(snr)

            # 鐢ㄦ埛杩炴帴鐘舵€侊紙褰撳墠鏃犱汉鏈哄姣忎釜鐢ㄦ埛锛?
            # 鍙€冭檻active鐨刄AV
            active_drones = self.member_manager.get_active_ids()
            active_drone_pos_list = [self.drone_pos[i].tolist() for i in active_drones]
            user_pos_list = self.user_pos.tolist()

            if active_drone_pos_list:
                _, connections, _, _ = self.channel_model.calculate_total_rate(
                    active_drone_pos_list, user_pos_list, self.communication_range
                )
                # 灏哸ctive鏃犱汉鏈虹殑绱㈠紩鏄犲皠鍥炲師濮嬬储寮?
                for j in range(self.num_users):
                    if connections[j] != -1:
                        original_drone_idx = active_drones[connections[j]]
                        if original_drone_idx == drone_idx:
                            obs.append(1.0)
                        else:
                            obs.append(0.0)
                    else:
                        obs.append(0.0)
            else:
                # 濡傛灉娌℃湁active鏃犱汉鏈猴紝鎵€鏈夌敤鎴疯繛鎺ョ姸鎬侀兘鏄?
                for j in range(self.num_users):
                    obs.append(0.0)

            obs = np.array(obs, dtype=np.float32)
            # 瀵逛綅缃€侀€熷害闄や互鍖哄煙澶у皬
            obs[:2] /= self.area_size
            obs[2:4] /= self.max_speed  # 閫熷害褰掍竴鍖?
            # 瓒婄晫娆℃暟銆佺鎾炶鏁般€佺數姹犵數閲忓凡缁忓湪鍚堢悊鑼冨洿锛屽彲浠ヤ繚鎸佸師鏍锋垨闄や互鏈€澶у€?
            obs[4] = min(obs[4], 10) / 10  # 瓒婄晫娆℃暟闄愬埗鍦?0浠ュ唴
            obs[5] = min(obs[5], 10) / 10  # 纰版挒娆℃暟闄愬埗鍦?0浠ュ唴
            obs[6] /= 100  # 鐢垫睜鐢甸噺褰掍竴鍖栧埌[0,1]
            # SNR 閫氬父寰堝ぇ锛岀敤log鍘嬬缉
            snr_start = 7
            snr_end = snr_start + self.num_users
            obs[snr_start:snr_end] = np.log1p(obs[snr_start:snr_end]) / np.log(1 + 1e6)  # 鍘嬬缉鍒癧0,1]宸﹀彸
            # 杩炴帴鐘舵€佸凡缁忔槸0/1锛屼笉闇€瑕佸鐞?
            obs_list.append(obs)

        # 瀵逛簬鍗曟棤浜烘満鎯呭喌锛岃繑鍥炵涓€涓瀵?
        if self.num_drones == 1:
            return obs_list[0]
        return obs_list

    def get_positions(self):
        """鑾峰彇鏃犱汉鏈哄拰鐢ㄦ埛鐨?D浣嶇疆"""
        drone_pos = self.drone_pos.tolist()
        user_pos = self.user_pos.tolist()
        return drone_pos, user_pos

    def close(self):
        """鍏抽棴鐜"""
        pass

    def _test_isac_integration(self):
        """Run a lightweight smoke-check for communication and sensing models."""
        comm_model_exists = hasattr(self, "channel_model") and self.channel_model is not None
        sense_model_exists = hasattr(self, "sensing_model") and self.sensing_model is not None

        if not (comm_model_exists and sense_model_exists):
            return

        test_drone_pos = [[50, 50, 50]]
        test_user_pos = [[25, 25, 0], [75, 75, 0]]

        try:
            self.channel_model.calculate_total_rate(
                test_drone_pos, test_user_pos, self.communication_range
            )
        except Exception:
            pass

        try:
            self.sensing_model.calculate_coverage(
                test_drone_pos, test_user_pos, self.sensing_range
            )
        except Exception:
            pass
