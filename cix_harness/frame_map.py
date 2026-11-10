"""Offline rigid frame construction from documented corresponding TCP points.

No robot connection, fitting to policy statistics, scale adjustment or motion.
The first three points define oriented frames; further points validate them.
"""
import argparse
import json
import math
from pathlib import Path

from .contract import vector, instruction
from .pi05_contract import multiply, quaternion, rotate


def sub(a,b):
    return [x-y for x,y in zip(a,b)]


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def basis(points):
    x=sub(points[1],points[0]);y=sub(points[2],points[0])
    length=math.sqrt(dot(x,x))
    if length<.02:raise ValueError('Reference baseline must be at least 2 cm')
    x=[v/length for v in x]
    y=sub(y,[dot(x,y)*v for v in x]);height=math.sqrt(dot(y,y))
    if height<.02:raise ValueError('Third reference must be at least 2 cm off the baseline')
    y=[v/height for v in y]
    return [x,y,cross(x,y)]  # columns


def matrix_quaternion(m):
    # Choose the largest quaternion component to avoid division near 180 degrees.
    candidates=[1+m[0][0]-m[1][1]-m[2][2],1-m[0][0]+m[1][1]-m[2][2],
                1-m[0][0]-m[1][1]+m[2][2],1+m[0][0]+m[1][1]+m[2][2]]
    i=max(range(4),key=lambda k:candidates[k]);s=2*math.sqrt(max(0,candidates[i]))
    if i==0:q=[s/4,(m[0][1]+m[1][0])/s,(m[0][2]+m[2][0])/s,(m[2][1]-m[1][2])/s]
    elif i==1:q=[(m[0][1]+m[1][0])/s,s/4,(m[1][2]+m[2][1])/s,(m[0][2]-m[2][0])/s]
    elif i==2:q=[(m[0][2]+m[2][0])/s,(m[1][2]+m[2][1])/s,s/4,(m[1][0]-m[0][1])/s]
    else:q=[(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s,s/4]
    return quaternion(q)


def build(data):
    evidence=instruction(data['reference_definition'])
    a=[vector(p,3,'base TCP point') for p in data['base_tcp_points_m']]
    b=[vector(p,3,'policy corresponding point') for p in data['policy_points_m']]
    if len(a)!=len(b) or len(a)<3:raise ValueError('At least three paired points required')
    tolerance=data.get('max_residual_m',.003)
    if type(tolerance) not in (float,int) or not math.isfinite(tolerance) or not 0<tolerance<=.02:
        raise ValueError('Residual tolerance must be in (0, 0.02] metres')
    u,v=basis(a),basis(b)
    m=[[sum(v[k][i]*u[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    q=matrix_quaternion(m);t=sub(b[0],rotate(q,a[0]))
    errors=[math.dist([x+y for x,y in zip(rotate(q,p),t)],r) for p,r in zip(a,b)]
    if max(errors)>tolerance:raise ValueError('Reference residual exceeds tolerance; no scale/reflection is fitted')
    sdk=quaternion(data['sdk_reference_xyzw']);policy=quaternion(data['policy_reference_xyzw'])
    combined=multiply(q,sdk)
    tool=multiply([-combined[0],-combined[1],-combined[2],combined[3]],policy)
    tcp=vector(data['tcp_offset_tool_m'],3,'TCP offset')
    profile={'commissioned':False,'policy_from_base_xyzw':q,
             'policy_from_base_translation_m':t,'tool_frame_offset_xyzw':tool,
             'orientation_quaternion_reference_xyzw':policy,'tcp_offset_tool_m':tcp,
             'notes':'UNVERIFIED READ-ONLY CANDIDATE. '+evidence}
    return {'policy_profile':profile,
            'bridge_candidate':{'commissioned':False,'policy_to_base_xyzw':[-q[0],-q[1],-q[2],q[3]],
                                'tcp_offset_tool_m':tcp},
            'point_residuals_m':errors,'extra_validation_points':max(0,len(a)-3),
            'reference_definition':evidence,
            'limitations':'Geometric consistency only. Does not establish camera alignment, policy transfer, TCP accuracy or motion permission.'}


def build_anchor(data):
    """Translation from one anchor, with rotation explicitly supplied, never fitted."""
    evidence=instruction(data['reference_definition'])
    q=quaternion(data['policy_from_base_xyzw'])
    base=vector(data['base_anchor_m'],3,'base anchor')
    policy=vector(data['policy_anchor_m'],3,'policy anchor')
    tcp=vector(data['tcp_offset_tool_m'],3,'TCP offset')
    tool=quaternion(data.get('tool_frame_offset_xyzw',[0,0,0,1]))
    t=sub(policy,rotate(q,base))
    return {'policy_profile':{'commissioned':False,'policy_from_base_xyzw':q,
                'policy_from_base_translation_m':t,'tcp_offset_tool_m':tcp,
                'tool_frame_offset_xyzw':tool,
                'notes':'UNVERIFIED SINGLE-ANCHOR HYPOTHESIS. '+evidence},
            'bridge_candidate':{'commissioned':False,'policy_to_base_xyzw':[-q[0],-q[1],-q[2],q[3]],
                                'tcp_offset_tool_m':tcp},
            'reference_definition':evidence,'base_anchor_m':base,'policy_anchor_m':policy,
            'limitations':'One point determines translation only with assumed rotation. No rotation, scale, camera alignment, transfer or motion validity established.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',required=True,help='Local JSON with paired TCP references and explicit tool conventions')
    p.add_argument('--mode',choices=['references','anchor'],default='references')
    p.add_argument('--output',required=True,help='New local report file; existing files are never overwritten')
    args=p.parse_args(argv)
    result=(build_anchor if args.mode=='anchor' else build)(json.loads(Path(args.references).read_text(encoding='utf-8')))
    with Path(args.output).open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'mode':args.mode,'point_residuals_m':result.get('point_residuals_m'),
                      'extra_validation_points':result.get('extra_validation_points'),'commissioned':False}))


if __name__=='__main__':main()
