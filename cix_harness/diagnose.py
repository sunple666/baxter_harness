"""Fixed-subtask VLA diagnostic. Preview by default; execute one arm-only action."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import time

from .__main__ import Journal
from .contract import ACTION_SPEC, action7, instruction
from .rollout import predict_window, validate_motion_model
from .rpc import HttpRpcClient


def run(args, log, robot=None, model=None):
    robot = robot or HttpRpcClient(args.robot_port)
    model = model or HttpRpcClient(args.vla_port)
    health = robot.call('healthz')
    meta = model.call('healthz') if args.manual_action is None else {'action_spec':ACTION_SPEC,'model':'manual'}
    if health.get('diagnostic_protocol') != 1:
        raise RuntimeError('Update and restart the robot bridge before diagnostics')
    if health.get('action_spec') != ACTION_SPEC or meta.get('action_spec') != ACTION_SPEC:
        raise ValueError('Action contract mismatch')
    if args.execute:
        if (health.get('armed') is not True or
            health.get('motion_scope') != 'supervised_cartesian_diagnostic' or
            health.get('motion_remaining') != 1):
            raise RuntimeError('Execution requires the dedicated armed diagnostic profile with budget 1')
        if args.manual_action is None:
            validate_motion_model(meta, health)
    elif health.get('armed') is not False:
        raise RuntimeError('Preview requires a read-only robot bridge')
    root = Path(__file__).resolve().parents[1]
    try:
        version = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=False).stdout.strip()
    except OSError:
        version = 'git_unavailable'
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).parent.glob('*.py')}
    log('start', route='diagnostic', mode='execute' if args.execute else 'preview',
        subtask=args.subtask, robot=health, model=meta,
        source_commit=version, source_sha256=sources,
        gripper_override='preserve', manual_action=args.manual_action)
    obs = robot.call('env.observe')
    log.camera(obs,'vla')
    if args.execute and obs.get('state',{}).get('gripping') is not False:
        raise RuntimeError('Air-motion diagnostic requires confirmed empty grasp telemetry')
    began = time.monotonic()
    if args.manual_action is None:
        actions = predict_window(model,obs,instruction(args.subtask),args.window,log)
        selected = actions[0]
    else:
        selected = action7(list(args.manual_action)+[1.0])
        log('manual_prediction',obs_id=obs['id'],action=selected)
    log('selected_action',obs_id=obs['id'],action=selected,
        seconds_since_observation=time.monotonic()-began,
        note='Only the first action is used. Gripper output is ignored during execution.')
    # Preview consumes the observation. Never preview and execute the same token.
    method = 'env.diagnostic_step' if args.execute else 'env.inspect'
    receipt = robot.call(method,(obs['id'],ACTION_SPEC,selected),timeout_s=30)
    log('applied' if args.execute else 'ik_preview',**receipt)
    if args.execute:
        if receipt.get('applied') is not True or receipt.get('obs_id') != obs['id'] or receipt.get('preserve_gripper') is not True:
            raise RuntimeError('Invalid diagnostic receipt; do not replay')
        after = robot.call('env.observe')
        log.camera(after,'after')
        if receipt.get('target_reached') is not True:
            raise RuntimeError('Diagnostic action did not reach target')
    log('finish',status='diagnostic_complete',task_success_claimed=False)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subtask',required=True,help='Fixed English instruction; no VLM is called')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--manual-action',nargs=6,type=float,metavar=('DX','DY','DZ','ROLL','PITCH','YAW'),
                   help='Bypass policy prediction; policy-frame metres/radians, gripper unchanged')
    p.add_argument('--window',type=int,default=16,choices=range(1,17))
    p.add_argument('--robot-port',type=int,default=9102)
    p.add_argument('--vla-port',type=int,default=9101)
    p.add_argument('--logs',default='runs')
    args = p.parse_args(argv)
    log = Journal(args.logs)
    try:
        return run(args,log)
    except (Exception,KeyboardInterrupt) as exc:
        log('stop',error_type=type(exc).__name__,reason=str(exc),replayed=False)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
