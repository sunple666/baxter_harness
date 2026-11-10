"""LeRobot Pi0.5-LIBERO on COREX, with strict streamed loading and real observations."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

from .contract import ACTION_SPEC, image_bytes, instruction
from .pi05_contract import physical_delta, quaternion, state8, tcp_offset
from .rpc import serve

REVISION = '9ba5431256f4136a01c224ab0307b0d3035cc92c'


def empty_policy(directory):
    from accelerate import init_empty_weights
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch

    cfg = PreTrainedConfig.from_pretrained(directory, local_files_only=True)
    if not isinstance(cfg, PI05Config):
        raise ValueError('Expected a Pi05 checkpoint configuration')
    cfg.device = 'cuda'
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    cfg.n_action_steps = 1
    cfg.validate_features()
    # Same upstream policy/model classes, but no full CPU model or randomly
    # initialized fallback if a checkpoint fails to load (upstream allows one).
    policy = PI05Policy.__new__(PI05Policy)
    PreTrainedPolicy.__init__(policy, cfg)
    policy.config = cfg
    policy.init_rtc_processor()
    with init_empty_weights(include_buffers=False):
        policy.model = PI05Pytorch(cfg, rtc_processor=policy.rtc_processor)
    return policy


def load_policy(directory):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    policy = empty_policy(directory)
    expected = policy.state_dict()
    with safe_open(str(directory / 'model.safetensors'), framework='pt', device='cpu') as weights:
        actual = set(weights.keys())
        if actual != set(expected):
            raise ValueError('Checkpoint key mismatch: missing=%s extra=%s' %
                             (sorted(set(expected)-actual)[:8], sorted(actual-set(expected))[:8]))
        for index, key in enumerate(sorted(actual)):
            value = weights.get_tensor(key)
            if value.shape != expected[key].shape:
                raise ValueError('Checkpoint shape mismatch: ' + key)
            set_module_tensor_to_device(policy, key, 'cuda:0', value=value, dtype=expected[key].dtype)
            del value
            if index % 100 == 0:
                print('Loaded tensor %d/%d' % (index, len(actual)), flush=True)
    del expected
    if any(p.is_meta for p in policy.parameters()):
        raise RuntimeError('Unloaded meta parameters remain')
    policy.to('cuda:0').eval()
    policy.reset()
    torch.cuda.synchronize()
    return policy


class Pi05:
    def __init__(self, model_dir, profile_file, log_dir):
        import torch
        from transformers import GemmaTokenizer
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        # Register the checkpoint's Pi05-specific state tokenizer step.
        import lerobot.policies.pi05.processor_pi05  # noqa: F401

        self.directory = Path(model_dir).resolve(strict=True)
        if (self.directory / 'verified-revision.txt').read_text().strip() != REVISION:
            raise ValueError('Run the pinned checkpoint integrity verifier first')
        self.profile = json.loads(Path(profile_file).read_text())
        self.profile_file = Path(profile_file)
        self.warmed = False
        quaternion(self.profile['policy_from_base_xyzw'])
        quaternion(self.profile['tool_frame_offset_xyzw'])
        tcp_offset(self.profile)
        self.logs = Path(log_dir)
        self.logs.mkdir(parents=True, exist_ok=True)
        if not torch.cuda.is_available():
            raise RuntimeError('COREX GPU is unavailable')
        self.torch = torch
        tokenizer_dir = self.directory / 'tokenizer'
        import hashlib
        if hashlib.sha256((tokenizer_dir / 'tokenizer.model').read_bytes()).hexdigest() != '8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6':
            raise ValueError('Tokenizer integrity failed')
        if not (tokenizer_dir / 'tokenizer_config.json').exists():
            tokenizer = GemmaTokenizer(vocab_file=str(tokenizer_dir / 'tokenizer.model'),
                                       add_bos_token=True, add_eos_token=False)
            tokenizer.save_pretrained(tokenizer_dir)
        self.policy = load_policy(self.directory)
        self.pre = PolicyProcessorPipeline.from_pretrained(
            str(self.directory), config_filename='policy_preprocessor.json',
            overrides={'tokenizer_processor': {'tokenizer_name': str(tokenizer_dir)},
                       'device_processor': {'device': 'cuda'}})
        self.post = PolicyProcessorPipeline.from_pretrained(
            str(self.directory), config_filename='policy_postprocessor.json',
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    def health(self):
        q = self.profile['policy_from_base_xyzw']
        return {'model': 'pi05-libero', 'checkpoint': 'lerobot/pi05-libero', 'revision': REVISION,
                'device': 'cuda:0', 'action_spec': ACTION_SPEC, 'requires_wrist': True,
                'max_action_window': 16, 'warmed': self.warmed,
                'motion_ready': self.profile.get('commissioned') is True,
                'motion_scope': self.profile.get('commissioning_scope', 'task_workspace'),
                'required_policy_to_base_xyzw': [-q[0], -q[1], -q[2], q[3]],
                'required_tcp_offset_tool_m': tcp_offset(self.profile),
                'adaptation': 'LIBERO OSC_POSE to physical xyz/rpy/open-fraction; bounded receding horizon',
                'allocated_gib': round(self.torch.cuda.memory_allocated()/2**30, 3)}

    def image_tensor(self, encoded):
        import numpy as np
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes(encoded))) as source:
            if source.width*source.height > 8_000_000:
                raise ValueError('Input image is too large')
            pixels = np.array(source.convert('RGB'), copy=True)
        return self.torch.from_numpy(pixels).permute(2, 0, 1).float()/255

    def predict(self, obs, options=None):
        count=(options or {}).get('action_window',1)
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError('Action window must be in 1..16')
        seed=(options or {}).get('diagnostic_seed')
        if seed is not None:
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError('Diagnostic seed must be an integer in 0..2**32-1')
            if self.profile.get('commissioned') is True:
                raise ValueError('Seeded diagnostic inference requires an uncommissioned profile')
        task = instruction(obs['task_descriptions'])
        if 'wrist_image' not in obs or 'state' not in obs:
            raise ValueError('Pi05 requires the real wrist image and measured endpoint/gripper state')
        state = state8(obs['state'], self.profile)
        batch = {'observation.images.image': self.image_tensor(obs['image']),
                 'observation.images.image2': self.image_tensor(obs['wrist_image']),
                 'observation.state': self.torch.tensor(state, dtype=self.torch.float32), 'task': task}
        start = time.monotonic()
        self.policy.reset()  # Never reuse a chunk from an older observation/task.
        with self.torch.inference_mode():
            prepared = self.pre(batch)
            kwargs = {}
            if seed is not None:
                # An isolated CPU generator leaves normal inference RNG untouched.
                # Upstream accepts explicit flow-matching noise via sample_actions.
                generator=self.torch.Generator(device='cpu').manual_seed(seed)
                cfg=self.policy.config
                kwargs['noise']=self.torch.randn(
                    (1,cfg.chunk_size,cfg.max_action_dim),generator=generator,
                    dtype=self.torch.float32).to('cuda:0')
            chunk = self.policy.predict_action_chunk(prepared, **kwargs)
            raw = self.post(chunk)[0].float().cpu().tolist()
        self.torch.cuda.synchronize()
        self.warmed = True
        if not raw or any(len(a) != 7 for a in raw):
            raise ValueError('Invalid Pi05 chunk shape')
        converted = [physical_delta(a) for a in raw]
        result = {'obs_id': obs['id'], 'action_spec': ACTION_SPEC, 'actions': converted[:count],
                  'model': 'pi05-libero', 'chunk_length': len(raw), 'seconds': round(time.monotonic()-start, 3),
                  'motion_ready': self.profile.get('commissioned') is True}
        normalized = prepared['observation.state'][0].float().cpu().tolist()
        result['state_diagnostic'] = {
            'state8':state,
            'normalized_state8':normalized,
            'outside_token_interval_indices':[i for i,v in enumerate(normalized) if v < -1 or v > 1],
            'note':'Distribution diagnostic only; values and checkpoint statistics are unchanged.'}
        record = {**result, 'task': task, 'state8': state, 'raw_libero_chunk': raw,
                  'profile': self.profile, 'checkpoint_revision': REVISION,
                  'diagnostic_seed': seed,
                  'normalized_state8': normalized,
                  'input_sha256': {key: hashlib.sha256(image_bytes(obs[key])).hexdigest()
                                   for key in ('image','wrist_image')},
                  'peak_allocated_gib': round(self.torch.cuda.max_memory_allocated()/2**30, 3)}
        # Observation IDs are untrusted strings; use an independent local filename.
        import uuid
        path = self.logs / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.json')
        path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2))
        result['trace_file'] = str(path)
        print(json.dumps(result), flush=True)
        return result

    def reload_profile(self):
        profile=json.loads(self.profile_file.read_text())
        quaternion(profile['policy_from_base_xyzw'])
        quaternion(profile['tool_frame_offset_xyzw'])
        tcp_offset(profile)
        self.profile=profile
        return self.health()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--logs', default='/opt/harness/runs/pi05')
    p.add_argument('--port', type=int, default=9101)
    args = p.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    model = Pi05(args.model, args.profile, args.logs)
    serve(args.port, {'healthz': model.health, 'vla.predict': model.predict,
                      'vla.reload_profile': model.reload_profile})


if __name__ == '__main__':
    main()
