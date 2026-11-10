import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cix_harness.baxter import Baxter, validate_config
from cix_harness.contract import ACTION_SPEC


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.robot = object.__new__(Baxter)
        self.robot.armed = True
        self.robot.calibrated_profile = True
        self.robot.cfg = {"max_observation_age_s": 2}
        self.robot.pending = ("frame", time.monotonic(), {"j": 0})
        self.robot.ready = Mock()
        self.robot.joints = Mock(return_value={"j": 0})
        self.robot.solve = self.robot.target = Mock()
        self.robot.move = Mock()
        self.robot.endpoint_position = Mock(return_value=[0.7,0.1,0.2])
        self.robot.arrival_error = Mock(return_value=(.001,.001))
        self.robot.grip = Mock()
        self.robot.grip.command_position.return_value = True

    def test_default_profile_rejects_motion(self):
        template = Path(__file__).resolve().parents[1] / "config/baxter.example.json"
        profile = json.loads(template.read_text())
        validate_config(profile, False)
        with self.assertRaises(ValueError):
            validate_config(profile, True)

    def test_telemetry_tolerates_small_clock_skew_without_renewing_replays(self):
        class Stamp:
            def __init__(self, value):
                self.value = value
            def to_sec(self):
                return self.value
            def __sub__(self, other):
                return Stamp(self.value - other.value)

        self.robot.ros = Mock()
        self.robot.ros.Time.now.return_value = Stamp(100)
        self.robot.state_stamps = {}
        with patch('cix_harness.baxter.time.monotonic', return_value=200):
            self.assertEqual(self.robot.telemetry_time(Stamp(100.03), 'joints'), 200)
            self.assertIsNone(self.robot.telemetry_time(Stamp(100.03), 'joints'))
            self.assertIsNone(self.robot.telemetry_time(Stamp(100.02), 'joints'))
            self.assertIsNone(self.robot.telemetry_time(Stamp(100.2), 'endpoint'))
            self.assertIsNone(self.robot.telemetry_time(Stamp(99.4), 'endpoint'))
            self.assertAlmostEqual(self.robot.telemetry_time(Stamp(99.8), 'endpoint'), 199.8)
            self.robot.cfg['robot_clock_offset_s']=.16
            self.assertEqual(self.robot.telemetry_time(Stamp(100.16), 'offset_probe'),200)
            self.assertIsNone(self.robot.telemetry_time(Stamp(100.16), 'offset_probe'))
            self.assertIsNone(self.robot.telemetry_time(Stamp(99.5), 'stale_with_offset'))

    def test_action_is_one_shot_even_after_success(self):
        self.robot.step("frame", ACTION_SPEC, [0]*7)
        with self.assertRaises(ValueError):
            self.robot.step("frame", ACTION_SPEC, [0]*7)
        self.robot.move.assert_called_once()

    def test_diagnostic_preserves_gripper_without_command_position(self):
        self.robot.cfg['commissioning_scope']='supervised_cartesian_diagnostic'
        self.robot.motion_remaining=1
        receipt=self.robot.diagnostic_step('frame',ACTION_SPEC,[0]*7)
        self.assertTrue(receipt['preserve_gripper'])
        self.robot.grip.command_position.assert_not_called()
        self.robot.move.assert_called_once()
        with self.assertRaises(RuntimeError):
            self.robot.diagnostic_step('frame',ACTION_SPEC,[0]*7)

    def test_read_only_bridge_projects_without_commanding_hardware(self):
        self.robot.armed = False
        with self.assertRaises(RuntimeError):
            self.robot.step("frame", ACTION_SPEC, [0]*7)
        self.assertFalse(self.robot.preview("frame", ACTION_SPEC, [0]*7)["applied"])
        self.robot.move.assert_not_called()
        self.robot.grip.command_position.assert_not_called()

    def test_read_only_observation_uses_fresh_gripper_telemetry(self):
        del self.robot.grip
        self.robot.armed = False
        self.robot.cfg["arm"] = "left"
        self.robot.camera = Mock()
        self.robot.camera.capture.return_value = ("image", time.monotonic(), 3)
        self.robot.limb = Mock()
        self.robot.arrival_error = Mock(return_value=(.001,.002))
        self.robot.limb.joint_names.return_value = ["j"]
        self.robot.limb.endpoint_pose.return_value = {"position": SimpleNamespace(x=0, y=0, z=0)}
        state = SimpleNamespace(position=97.0, gripping=2, STATE_TRUE=1, STATE_UNKNOWN=2)
        self.robot.effector_state = (time.monotonic(), state)
        self.assertIsNone(self.robot.observe()["state"]["gripping"])
        self.robot.effector_state = (time.monotonic() - 1, state)
        with self.assertRaisesRegex(RuntimeError, "gripper telemetry"):
            self.robot.observe()
        self.assertIsNone(self.robot.pending)

    def test_expired_and_changed_pose_and_invalid_values_are_rejected(self):
        for captured, joints, action in [(0, {"j": 0}, [0]*7),
                                          (time.monotonic(), {"j": 1}, [0]*7),
                                          (time.monotonic(), {"j": 0}, [float("nan")]*7)]:
            self.robot.pending = ("frame", captured, joints)
            with self.assertRaises((ValueError, RuntimeError)):
                self.robot.step("frame", ACTION_SPEC, action)
        self.robot.move.assert_not_called()

    def test_uncommissioned_ik_inspection_never_authorizes_motion(self):
        self.robot.calibrated_profile = False
        self.robot.ik = Mock()
        p = SimpleNamespace(x=0.7,y=0.1,z=0.2)
        q = SimpleNamespace(x=0,y=0,z=0,w=1)
        self.robot.target = Mock(return_value=SimpleNamespace(pose=SimpleNamespace(position=p,orientation=q)))
        self.robot.solve = Mock(return_value={'j':0.01})
        result = self.robot.inspect_action('frame', ACTION_SPEC, [0]*7)
        self.assertFalse(result['motion_allowed'])
        self.assertFalse(result['workspace_verified'])
        self.robot.target.assert_called_once_with([0]*7, enforce_workspace=False)
        self.robot.move.assert_not_called()
        self.robot.grip.command_position.assert_not_called()
        with self.assertRaises(RuntimeError):
            self.robot.step('frame', ACTION_SPEC, [0]*7)

    def test_translation_limit_cannot_be_raised_above_hard_cap(self):
        with self.assertRaises(ValueError):
            validate_config({'arm':'left','max_translation_m':0.02}, False)

    def test_five_cm_clearance_scope_requires_single_budget_and_keeps_cap(self):
        cfg = {'arm':'left','commissioning_scope':'local_clearance_single_step',
               'motion_budget':1,'max_translation_m':0.05}
        validate_config(cfg, False)
        for key,value in [('motion_budget',2),('max_translation_m',0.051)]:
            with self.assertRaises(ValueError):
                validate_config(dict(cfg, **{key:value}), False)

    def test_robot_side_budget_blocks_second_action_even_with_fresh_id(self):
        self.robot.motion_remaining = 1
        self.robot.step('frame', ACTION_SPEC, [0]*7)
        self.robot.pending = ('second', time.monotonic(), {'j':0})
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            self.robot.step('second', ACTION_SPEC, [0]*7)
        self.robot.move.assert_called_once()

    def test_clearance_trajectory_limits_speed_and_stops_tracking_error(self):
        self.robot.cfg['commissioning_scope'] = 'local_clearance_single_step'
        self.robot.inside = Mock()
        clock = [0.0]
        measured = {'j':0.0}
        samples = [(0.0,0.0)]
        self.robot.joints = lambda: dict(measured)
        self.robot.limb = Mock()
        def follow(command):
            measured.update(command)
            samples.append((clock[0],command['j']))
        self.robot.limb.set_joint_positions.side_effect = follow
        with patch('cix_harness.baxter.time.monotonic', side_effect=lambda:clock[0]), \
             patch('cix_harness.baxter.time.sleep', side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)):
            self.robot.arrival_error = Mock(return_value=(.001,.002))
            Baxter.move(self.robot, {'j':.1}, Mock())
            self.assertLess(abs(measured['j']-.1),.002)
            for (ta,a),(tb,b) in zip(samples,samples[1:]):
                if tb > ta:
                    self.assertLessEqual(abs(b-a)/(tb-ta),.06001)
            measured['j'] = 0
            self.robot.limb.set_joint_positions.side_effect = None
            with self.assertRaisesRegex(RuntimeError,'tracking error'):
                Baxter.move(self.robot, {'j':.1}, Mock())

    def test_arrival_checks_cartesian_error_in_addition_to_joint_error(self):
        del self.robot.arrival_error
        self.robot.limb = Mock()
        q = SimpleNamespace(x=0,y=0,z=0,w=1)
        self.robot.limb.endpoint_pose.return_value = {
            'position':SimpleNamespace(x=.7,y=.1,z=.2),'orientation':q}
        target = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=.71,y=.1,z=.2),orientation=q))
        distance, angle = self.robot.arrival_error(target)
        self.assertAlmostEqual(distance,.01)
        self.assertAlmostEqual(angle,0)

    def test_supervised_sequence_enforces_cumulative_radius_and_budget(self):
        cfg={'arm':'left','commissioning_scope':'supervised_cartesian_diagnostic',
             'motion_budget':5,'diagnostic_origin_m':[0,0,0],'diagnostic_radius_m':.18,
             'ik_solver':'kdl_continuous','max_translation_m':.05,
             'workspace_min_m':[-.18]*3,'workspace_max_m':[.18]*3}
        validate_config(cfg,False)
        self.robot.cfg=cfg
        self.robot.inside([0,0,.15])
        with self.assertRaisesRegex(ValueError,'cumulative'):
            self.robot.inside([.15,.15,0])
        for key,value in [('motion_budget',6),('diagnostic_radius_m',.21),('ik_solver','baxter_service')]:
            with self.assertRaises(ValueError):
                validate_config(dict(cfg,**{key:value}),False)

    def test_action_window_preserves_original_frame_age_and_consumes_budget(self):
        self.robot.cfg['commissioning_scope']='supervised_policy_task'
        self.robot.motion_remaining=2
        result=self.robot.step_chunk('frame',ACTION_SPEC,[[0]*7,[0]*7])
        self.assertEqual(result['applied_actions'],2)
        self.assertEqual(self.robot.motion_remaining,0)
        self.assertIsNone(self.robot.pending)
        with self.assertRaises(ValueError):
            self.robot.step_chunk('frame',ACTION_SPEC,[[0]*7])
        self.assertEqual(self.robot.move.call_count,2)

    def test_action_window_validates_all_values_before_first_motor_command(self):
        self.robot.cfg['commissioning_scope']='supervised_policy_task'
        self.robot.motion_remaining=2
        with self.assertRaises(ValueError):
            self.robot.step_chunk('frame',ACTION_SPEC,[[0]*7,[float('nan')]*7])
        self.robot.move.assert_not_called()

    def test_segment_streams_internal_points_and_settles_only_at_end(self):
        self.robot.cfg.update(max_translation_m=.12,joint_speed_rad_s=.12)
        self.robot.inside=Mock()
        self.robot.limb=Mock()
        measured={'j':0.};clock=[0.];samples=[]
        self.robot.joints=lambda:dict(measured)
        def follow(command):
            measured.update(command);samples.append((clock[0],command['j']))
        self.robot.limb.set_joint_positions.side_effect=follow
        self.robot.arrival_error=Mock(return_value=(.001,.001))
        with patch('cix_harness.baxter.time.monotonic',side_effect=lambda:clock[0]), \
             patch('cix_harness.baxter.time.sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)):
            self.robot.move_segment({'j':0.},[({'j':.1},Mock()),({'j':.2},Mock())],[.7,.1,.2])
        self.assertAlmostEqual(measured['j'],.2)
        self.assertGreaterEqual(clock[0],2*1.875*.1/.12+.2)
        for (ta,a),(tb,b) in zip(samples,samples[1:]):
            self.assertLessEqual(abs(b-a)/(tb-ta),.12001)
