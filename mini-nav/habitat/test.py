import habitat_sim
import numpy as np
import plotly.express as px

# 配置场景
scene_path = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.enable_physics = False

# 配置 agent
agent_cfg = habitat_sim.agent.AgentConfiguration()
rgb_sensor_spec = habitat_sim.CameraSensorSpec()
rgb_sensor_spec.uuid = "color_sensor"
rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor_spec.resolution = [256, 256]
rgb_sensor_spec.position = [0.0, 1.5, 0.0]

agent_cfg.sensor_specifications = [rgb_sensor_spec]

# 创建 simulator 实例
cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

# 初始化 agent
agent = sim.initialize_agent(0)

# 设置 agent 初始位置
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])
agent.set_state(agent_state)

state = agent.get_state()
print("位置:", state.position)
print("旋转四元数:", state.rotation)

observations = sim.get_sensor_observations()
rgb_image = observations["color_sensor"]  # numpy array
print("RGB shape:", rgb_image.shape)


# 假设 rgb_image 已经被定义（例如一个 numpy array 或 PIL Image）
fig = px.imshow(rgb_image)

# 隐藏坐标轴（等同于 plt.axis("off")）
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# 去除多余的边距，使图片填满画布
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_showscale=False,  # 如果是灰度图或带colorbar，这行可以隐藏色条
)

# 输出成 PNG 图片
fig.write_html("outputs/output.html")
