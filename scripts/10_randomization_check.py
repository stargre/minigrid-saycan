from environments.saycan_grid import SayCanGridEnv

for seed in range(10):
    env = SayCanGridEnv()
    env.reset(seed=seed)

    base = env.unwrapped

    print(
        f"seed={seed:02d} | "
        f"ball_side={base.ball_side:5s} | "
        f"ball={base.ball_pos} | "
        f"door={base.door_pos} | "
        f"target={base.target_pos}"
    )

    env.close()
