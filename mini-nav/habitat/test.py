import habitat_sim

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

color_sensor = habitat_sim.CameraSensorSpec()
color_sensor.uuid = "color_sensor"
color_sensor.sensor_type = habitat_sim.SensorType.COLOR
color_sensor.resolution = [480, 640]
color_sensor.position = [0.0, 1.5, 0.0]

agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [color_sensor]

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

obs = sim.get_sensor_observations()
print("obs keys:", obs.keys())
rgb = obs["color_sensor"]
print(rgb.shape)
