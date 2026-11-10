# Copyright 2026 The RPent Authors. Licensed under Apache-2.0.
# Adapted from robots/franka/tools.py::vla_grasp. See third_party/RPent-LICENSE.
# Baxter receives bounded policy windows, with a one-use observation ID.
from .contract import ACTION_SPEC, action7, vector


def validate_motion_model(policy, robot):
    if policy.get("motion_ready", True) is not True:
        raise RuntimeError("The selected policy's embodiment/frame adapter is not commissioned")
    expected_tcp = vector(policy.get('required_tcp_offset_tool_m',[0,0,0]),3,'policy TCP offset')
    actual_tcp = vector(robot.get('tcp_offset_tool_m',[0,0,0]),3,'robot TCP offset')
    if any(abs(a-b)>1e-8 for a,b in zip(expected_tcp,actual_tcp)):
        raise RuntimeError('Robot and policy TCP offsets differ')
    expected = policy.get("required_policy_to_base_xyzw")
    if expected is not None:
        expected = vector(expected, 4, "policy rotation")
        actual = vector(robot.get("policy_to_base_xyzw"), 4, "robot rotation")
        if (abs(sum(a*a for a in actual)-1) > 1e-5 or
                abs(sum(a*a for a in expected)-1) > 1e-5 or
                abs(abs(sum(a*b for a,b in zip(actual, expected)))-1) > 1e-5):
            raise RuntimeError("Robot and policy coordinate rotations differ")


def predict(model, obs, subtask):
    return predict_window(model,obs,subtask,1)[0]


def predict_window(model, obs, subtask, count, record=None):
    inputs = {"id": obs["id"], "image": obs["image"], "task_descriptions": subtask}
    for key in ("state", "wrist_image"):
        if key in obs:
            inputs[key] = obs[key]
    prediction = model.call("vla.predict", (inputs, {"mode": "eval",'action_window':count}))
    if prediction["obs_id"] != obs["id"] or prediction["action_spec"] != ACTION_SPEC:
        raise ValueError("Prediction does not match the current observation/action contract")
    actions = prediction["actions"]
    if not isinstance(actions, list) or len(actions) != count:
        raise ValueError("The policy adapter returned the wrong action-window length")
    actions = [action7(a) for a in actions]
    if record is not None:
        record('prediction', obs_id=obs['id'], actions=actions, subtask=subtask,
               trace_file=prediction.get('trace_file'), model=prediction.get('model'),
               state_diagnostic=prediction.get('state_diagnostic'),seconds=prediction.get('seconds'))
    return actions


def rollout(robot, model, subtask, steps, record, action_window=1, monitor=None, execution_mode='waypoints'):
    applied=0
    while applied < steps:
        count=min(action_window,steps-applied)
        if execution_mode=='segment':
            remaining=robot.call('healthz').get('motion_remaining')
            if remaining is not None:
                if remaining<=0:break
                count=min(count,remaining)
        # Planning can be slow on the CPU. Fetch again before every motor prediction.
        obs = robot.call("env.observe")
        record.camera(obs, "vla")
        if monitor:
            monitor.observe(obs.get('state',{}))
            monitor.require_progress()
        actions = predict_window(model, obs, subtask,count,record=record)
        # A transport error may mean the command already ran. Never replay it.
        if execution_mode=='segment':
            receipt=robot.call('env.step_segment',(obs['id'],ACTION_SPEC,actions),timeout_s=45)
        else:
            receipt = (robot.call('env.step_chunk',(obs['id'],ACTION_SPEC,actions),timeout_s=40) if count>1 else
                       robot.call("env.step", (obs["id"], ACTION_SPEC, actions[0]), timeout_s=12))
        if not isinstance(receipt, dict) or receipt.get("applied") is not True or receipt.get("obs_id") != obs["id"]:
            raise RuntimeError("Robot did not acknowledge this action")
        record("applied", **receipt)
        if receipt.get('target_reached') is False and execution_mode!='segment':
            raise RuntimeError('Action was applied but the final measured pose is outside arrival tolerance')
        if receipt.get('target_reached') is False:
            record('tracking_feedback',obs_id=obs['id'],reason='Reobserve after segment endpoint error',
                   position_error_m=receipt.get('position_error_m'))
        used=receipt.get('applied_actions',1)
        if type(used) is not int or not 1 <= used <= count:
            raise RuntimeError('Invalid window execution count')
        applied+=used
        if monitor:
            if execution_mode=='segment':
                monitor.applied(receipt,actions[used-1])
            else:
                for action, item in zip(actions, receipt.get('receipts', [receipt])):
                    monitor.applied(item, action)
            monitor.require_progress()
        if receipt.get('target_reached') is False:
            break
        if actions[used-1][6] < .5 or receipt.get('reobserve_reason') == 'gripper_transition':
            break
        if used<count and receipt.get('reobserve_reason') not in ('observation_deadline','path_budget','direction_change','ik_horizon'):
            break
    return applied
