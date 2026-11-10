import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cix_harness.contract import ACTION_SPEC
from cix_harness.diagnose import run
from cix_harness.baxter import Baxter


class DiagnosticTest(unittest.TestCase):
    def setup_run(self, execute=False):
        args = SimpleNamespace(robot_port=1,vla_port=2,execute=execute,
            subtask='pick up the cyan block',manual_action=[.005,0,0,0,0,0],window=16)
        robot = Mock()
        health = {'diagnostic_protocol':1,'action_spec':ACTION_SPEC,'armed':execute,
                  'motion_scope':'supervised_cartesian_diagnostic','motion_remaining':1}
        def call(method,*a,**kw):
            if method=='healthz':return health
            if method=='env.observe':return {'id':'fresh','state':{'gripping':False},'image':'fixture'}
            if method=='env.inspect':return {'applied':False,'obs_id':'fresh'}
            if method=='env.diagnostic_step':return {'applied':True,'obs_id':'fresh',
                'preserve_gripper':True,'target_reached':True}
            raise AssertionError(method)
        robot.call.side_effect=call
        return args,robot,Mock(),Mock(),health

    def test_manual_preview_has_no_policy_or_motor_call(self):
        args,robot,model,log,_ = self.setup_run()
        self.assertEqual(run(args,log,robot,model),0)
        model.call.assert_not_called()
        self.assertEqual([c.args[0] for c in robot.call.call_args_list],
                         ['healthz','env.observe','env.inspect'])

    def test_execution_is_once_with_no_prepare_or_replay(self):
        args,robot,model,log,_ = self.setup_run(True)
        self.assertEqual(run(args,log,robot,model),0)
        self.assertEqual([c.args[0] for c in robot.call.call_args_list].count('env.diagnostic_step'),1)
        def lost(method,*a,**kw):
            if method=='env.diagnostic_step':raise TimeoutError('ack lost')
            return {'healthz':{'diagnostic_protocol':1,'action_spec':ACTION_SPEC,'armed':True,
                'motion_scope':'supervised_cartesian_diagnostic','motion_remaining':1},
                'env.observe':{'id':'fresh','state':{'gripping':False},'image':'fixture'}}[method]
        robot.reset_mock();robot.call.side_effect=lost
        with self.assertRaises(TimeoutError):run(args,log,robot,model)
        self.assertEqual([c.args[0] for c in robot.call.call_args_list].count('env.diagnostic_step'),1)

    def test_wrong_scope_stops_before_observation(self):
        args,robot,model,log,health = self.setup_run(True)
        health['motion_scope']='supervised_policy_task'
        with self.assertRaises(RuntimeError):run(args,log,robot,model)
        self.assertEqual(robot.call.call_count,1)

    def test_model_preview_records_window_and_does_not_execute(self):
        args,robot,model,log,_ = self.setup_run()
        args.manual_action=None
        def call(method,*a,**kw):
            if method=='healthz':return {'action_spec':ACTION_SPEC}
            if method=='vla.predict':return {'obs_id':'fresh','action_spec':ACTION_SPEC,
                'actions':[[.001,0,0,0,0,0,1]]*16,'trace_file':'trace.json'}
            raise AssertionError(method)
        model.call.side_effect=call
        self.assertEqual(run(args,log,robot,model),0)
        self.assertEqual([c.args[0] for c in model.call.call_args_list],['healthz','vla.predict'])
        self.assertEqual(robot.call.call_args.args[0],'env.inspect')
        prediction=[c for c in log.call_args_list if c.args[0]=='prediction'][0]
        self.assertEqual(len(prediction.kwargs['actions']),16)

    def test_bridge_diagnostic_scope_and_budget(self):
        robot=object.__new__(Baxter)
        robot.cfg={'commissioning_scope':'supervised_cartesian_diagnostic'}
        robot.motion_remaining=1;robot.step=Mock()
        robot.diagnostic_step('fresh',ACTION_SPEC,[0]*7)
        robot.step.assert_called_once_with('fresh',ACTION_SPEC,[0]*7,preserve_gripper=True)
        robot.motion_remaining=0
        with self.assertRaises(RuntimeError):robot.diagnostic_step('fresh',ACTION_SPEC,[0]*7)
