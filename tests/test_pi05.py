import math
import unittest

from cix_harness.pi05_contract import physical_delta, state8, rotvec, tcp_position, endpoint_for_tcp_delta, tcp_offset
from cix_harness.rollout import validate_motion_model


class Pi05ContractTest(unittest.TestCase):
    def test_tcp_rotation_preserves_grasp_point_and_requested_delta(self):
        position=[.7,.1,.2];before=[0,0,0,1]
        after=[0,math.sin(math.pi/4),0,math.cos(math.pi/4)]
        offset=[0,0,.12];delta=[.01,-.02,.03]
        endpoint=endpoint_for_tcp_delta(position,before,after,delta,offset)
        start=tcp_position(position,before,offset)
        end=tcp_position(endpoint,after,offset)
        for a,b,d in zip(start,end,delta):self.assertAlmostEqual(b-a,d)

    def test_tcp_state_uses_rotated_offset_before_policy_transform(self):
        profile={'policy_from_base_xyzw':[0,0,0,1],
                 'policy_from_base_translation_m':[0,0,0],
                 'tool_frame_offset_xyzw':[0,0,0,1], 'tcp_offset_tool_m':[0,0,.12]}
        state={'eef_m':[.7,.1,.2],'eef_xyzw':[0,1,0,0],'gripper_percent':50}
        result=state8(state,profile)
        for a,b in zip(result[:3],[.7,.1,.08]):self.assertAlmostEqual(a,b)
        with self.assertRaises(ValueError):tcp_offset({'tcp_offset_tool_m':[0,0,float('nan')]})

    def test_tcp_mismatch_blocks_execution_including_older_bridge(self):
        p={'motion_ready':True,'required_tcp_offset_tool_m':[0,0,.12]}
        with self.assertRaisesRegex(RuntimeError,'TCP'):validate_motion_model(p,{})
        validate_motion_model(p,{'tcp_offset_tool_m':[0,0,.12]})
        with self.assertRaisesRegex(RuntimeError,'TCP'):
            validate_motion_model({'motion_ready':True},{'tcp_offset_tool_m':[0,0,.12]})
    def test_libero_rotation_vector_preserves_branch_above_pi(self):
        angle=math.pi+.04
        result=rotvec([math.sin(angle/2),0,0,math.cos(angle/2)])
        self.assertAlmostEqual(result[0],angle)
    def test_uncommissioned_or_mismatched_policy_cannot_execute(self):
        with self.assertRaises(RuntimeError):
            validate_motion_model({'motion_ready':False}, {})
        with self.assertRaises(RuntimeError):
            validate_motion_model({'motion_ready':True,'required_policy_to_base_xyzw':[0,0,0,1]},
                                  {'policy_to_base_xyzw':[0,0,1,0]})
        validate_motion_model({'motion_ready':True,'required_policy_to_base_xyzw':[0,0,0,1]},
                              {'policy_to_base_xyzw':[0,0,0,-1]})

    def test_normalized_actions_have_physical_units_and_correct_gripper_polarity(self):
        result = physical_delta([1, -1, 0.5, 0, 0, 0.2, 1])
        self.assertEqual(result[:3], [0.05, -0.05, 0.025])
        self.assertAlmostEqual(result[5], 0.1)
        self.assertEqual(result[6], 0)
        self.assertEqual(physical_delta([0, 0, 0, 0, 0, 0, -1])[6], 1)

    def test_axis_angle_is_converted_instead_of_relabelled_as_euler(self):
        result = physical_delta([0, 0, 0, 0.4, 0.4, 0, 0])
        self.assertGreater(abs(result[5]), 0.01)
        with self.assertRaises(ValueError):
            physical_delta([float('nan')]*7)

    def test_state_uses_measured_pose_and_explicit_frame_transform(self):
        profile = {'policy_from_base_xyzw': [0,0,math.sin(math.pi/4),math.cos(math.pi/4)],
                   'policy_from_base_translation_m':[1,2,3], 'tool_frame_offset_xyzw':[0,0,0,1]}
        state = {'eef_m':[1,0,0], 'eef_xyzw':[0,0,0,1], 'gripper_percent':50}
        result = state8(state, profile)
        for actual, expected in zip(result, [1,3,3,0,0,math.pi/2,.02,-.02]):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaises(KeyError):
            state8({'eef_m':[0,0,0]}, profile)
