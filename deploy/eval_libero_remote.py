"""Native LIBERO evaluation with per-episode records, videos and process logs.

Requires a simulation-only RPC exposing healthz and predict (native action chunk).
Use the same simulator and checkpoint environment as the experiment; no robot RPC.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def initial_record(task_id, index, task):
    return dict(task_id=task_id, initial_state_index=index, environment_seed=index,
                policy_seed=None, policy_rng_note='Persistent server RNG; not independently reset.',
                task=task, success=None, status='running', steps=0, inference_seconds=[], video=None)


def evaluate(args):
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from cix_harness.rpc import HttpRpcClient

    out = Path(args.output)
    task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task_id)
    rpc = HttpRpcClient(args.port)
    health = rpc.call('healthz')
    save_json(out / 'policy-health.json', health)
    if health.get('robot_commands') is not False or health.get('mode') != 'native_simulation_only':
        raise ValueError('A simulation-only policy service is required')
    init = torch.load(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file,
                      map_location='cpu', weights_only=False)
    # Initial states must come from the recorded, hash-verified upstream resource set.
    def encode(image):
        buffer = io.BytesIO()
        Image.fromarray(image[::-1, ::-1]).save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

    rows = []
    for index in range(args.episodes):
        stem = 'task%02d-init%02d' % (args.task_id, index)
        row = initial_record(args.task_id, index, task.language)
        row.update(suite=args.suite, max_steps=args.max_steps, full_log='console.log',
                   trajectory=stem + '.jsonl')
        env = writer = None
        started = time.monotonic()
        with (out / (stem + '.jsonl')).open('w', encoding='utf-8') as trajectory:
            try:
                np.random.seed(index)
                env = OffScreenRenderEnv(
                    bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                    camera_heights=256, camera_widths=256)
                env.seed(index)
                env.reset()
                obs = env.set_init_state(init[index])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                writer = imageio.get_writer(str(out / (stem + '.mp4')), fps=10)
                row['video'] = stem + '.mp4'

                def frame():
                    writer.append_data(np.concatenate([obs['agentview_image'][::-1, ::-1],
                                                       obs['robot0_eye_in_hand_image'][::-1, ::-1]], axis=1))
                frame()
                row['success'] = False
                while row['steps'] < args.max_steps:
                    state = np.concatenate([obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']),
                                            obs['robot0_gripper_qpos']]).astype(np.float32).tolist()
                    request = dict(state8=state, task=task.language, image=encode(obs['agentview_image']),
                                   wrist_image=encode(obs['robot0_eye_in_hand_image']))
                    start_call = time.monotonic()
                    prediction = rpc.call('predict', args=[request])
                    row['inference_seconds'].append(prediction['seconds'])
                    trajectory.write(json.dumps(dict(step=row['steps'], state8=state,
                        rpc_wall_seconds=time.monotonic()-start_call, prediction=prediction)) + '\n')
                    trajectory.flush()
                    actions = np.asarray(prediction['actions'], dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                        raise ValueError('Invalid native action chunk')
                    if not len(actions):
                        raise ValueError('Empty native action chunk')
                    done = False
                    for action in actions[:10]:
                        if row['steps'] >= args.max_steps:
                            break
                        obs, _, done, _ = env.step(np.clip(action, -1, 1))
                        row['steps'] += 1
                        row['success'] = bool(env.check_success())
                        if row['steps'] % 2 == 0 or row['success'] or done:
                            frame()
                        if row['success'] or done:
                            break
                    if row['success'] or done:
                        row['status'] = 'success' if row['success'] else 'environment_done'
                        break
                else:
                    row['status'] = 'time_limit'
                frame()  # Retain the terminal observation even for an odd final step.
            except Exception:
                row.update(status='error', success=None, error=traceback.format_exc())
                traceback.print_exc()
            finally:
                for name, resource in [('video', writer), ('environment', env)]:
                    if resource is not None:
                        try:
                            resource.close()
                        except Exception:
                            row.setdefault('cleanup_errors', {})[name] = traceback.format_exc()
                row['wall_seconds'] = time.monotonic()-started
                timings = row['inference_seconds']
                row['mean_inference_seconds'] = sum(timings)/len(timings) if timings else None
                save_json(out / (stem + '.json'), row)
                rows.append(row)
                save_json(out / 'results.json', rows)
                print('EPISODE_RESULT', json.dumps(row, ensure_ascii=False), flush=True)
    return int(any(r['status'] == 'error' or r.get('cleanup_errors') for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='libero_10')
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--port', type=int, default=9103)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.task_id < 0:
        parser.error('Episode count and horizon must be positive; task ID must be nonnegative')
    if args.worker:
        return evaluate(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    revision = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent.parent), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
    save_json(out / 'run-config.json', dict(vars(args), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        code_revision=revision.stdout.strip() if revision.returncode == 0 else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with (out / 'console.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                                stdout=log, stderr=subprocess.STDOUT)
    save_json(out / 'process-result.json', dict(exit_code=result.returncode,
        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
