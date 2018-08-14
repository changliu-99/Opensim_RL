def process_observation(env): #attempt to correct for the pelvis position
    for i in env.get_state_desc()['body_pos']:
        if i != 'pelvis':
            env.get_state_desc()['body_pos'][i][0] = env.get_state_desc()['body_pos'][i][0] - env.get_state_desc()['body_pos']['pelvis'][0]
    for i in env.get_state_desc()['joint_pos']:
        if i != 'ground_pelvis':
            env.get_state_desc()['joint_pos'][i][0] = env.get_state_desc()['joint_pos'][i][0] - env.get_state_desc()['joint_pos']['pelvis'][0]
    return observation_matrix
