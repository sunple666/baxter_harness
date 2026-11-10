# 板端 π0.5 + Baxter 单动作诊断

## 构造候选坐标映射（纯离线）

`frame_map.camera_yaw_candidate` 可按真实相机与策略相机的图像水平轴提出保留重力方向的 yaw 候选。必须显式声明策略图像是否经过水平翻转；LIBERO 渲染约定、LeRobot 环境图像翻转与真实相机光学系不能混为一谈。候选不是完整手眼标定。若采用约 180 度的世界轴转换，工具局部约定也必须一致处理；不要只反转动作 Y 或单独旋转状态。

若目前只有一个参考点，可用 `frame_map --mode anchor`：输入 base_anchor_m、policy_anchor_m、明确假设的 policy_from_base_xyzw、tcp_offset_tool_m 和 reference_definition，工具仅计算 t=policy_anchor-R*base_anchor，绝不声称由一个点标定旋转。tool_frame_offset_xyzw 可单独指定，默认不额外旋转。已知真实距离保持不变，不做缩放。LIBERO 不同场景的桌高与原点可能不同，必须声明选定场景，不能把单一桌高当成整个混合检查点的通用值。

`python3 -m cix_harness.frame_map --references config/frame-references.local.json --output runs/frame-candidate.local.json`

先复制 config/frame-references.example.json，填入有明确来源的参考数据；模板 null 故意不能直接运行。base_tcp_points_m 必须是实际 TCP 的 base 坐标，不是 SDK endpoint，不可直接混用相机坐标或贴桌末端记录。policy_points_m 是同一参考点在明确定义的策略任务坐标系中的坐标，不能仅为使归一化状态落入范围而构造。

至少三点：第一点为参考、第二点定义方向、第三点不共线；前两点间距与第三点到基线距离均需至少 2 cm，减少病态。前三点构造右手刚体变换，不调整尺度或拟合镜像。建议再提供独立验证点；输出包含全部残差，默认超过 3 mm 拒绝，但这个阈值只是参考数据一致性检查，不是 Baxter 精度保证。仅三点或共面点不能排除所有错误对应和镜像语义。

sdk_reference_xyzw 是参考姿态下实测 SDK 工具朝向；policy_reference_xyzw 是同一物理姿态在已核验策略工具约定中的表示。工具旋转由这两个约定和位置坐标变换计算，不搜索让模型输出更好看的角度。没有确认策略约定时，不能把训练均值当作目标姿态。

输出是报告，policy_profile 和 bridge_candidate 分别给出互逆位置旋转及相同 TCP；两者都 commissioned=false。它不会连接设备、更新活动配置或执行动作，也不会自动校准相机。先审查候选，再用录制观测复算状态与动作；几何自洽不等于零样本迁移成立。

## TCP 适配（默认保持旧行为）

策略 profile 和机器人桥配置现在都支持可选 `tcp_offset_tool_m`：SDK endpoint 到实际抓取参考点的向量，单位米，表达在 SDK 工具坐标系。缺省为零，不填猜测值。模型 state8 使用该点；控制桥按目标姿态反算 SDK endpoint，使旋转时仍满足该 TCP 的平移要求。它不是 `tool_frame_offset_xyzw`（姿态约定），也不是 camera→base 外参。

两端 offset 必须一致，否则拒绝策略执行；旧桥不能执行带非零 offset 的新策略。补偿导致 SDK endpoint 超过既有单步位移上限时直接拒绝，不放宽限制。仍需核验全臂、指尖及桌面空间。字段存在或数值落在训练范围内，不代表完成标定。

这项改动只补齐适配能力，不会自动确定策略坐标原点、朝向或正确相机布局；现有零平移与旧虚拟映射的只读对照均有输入饱和，不能直接提升为可信运动配置。现场仍保持只读，待确认实测工具偏移和坐标定义后再建立新 profile。

本次新增入口只做观测、预测、预览或一次机械臂动作，不运行 Qwen，不判定抓取成功。默认只读。执行时忽略策略的夹爪输出，不发送 command_position；异常处理仍可能调用夹爪 stop。它不是自主抓取入口。

## 两端部署

将本次修改的 cix_harness 目录同步到板子和控制主机的现有工程目录，保留各自模型、vendor、真实配置和 runs。不是 Git checkout 的现场目录不要直接 git pull。重启相关服务才会加载新代码。

涉及文件：diagnose.py、baxter.py、pi05.py、rollout.py。机器人桥和 π0.5 服务均需加载新版。旧入口保留。不要把示例配置当作真实标定，也不要覆盖学长正在修改的原语配置。

控制电脑按既有流程加载 ROS Noetic 与工作区，使用已核验的 VLA 配置启动只读桥（不加 --arm-motion），不能使用原语 scope。板子恢复现有 π0.5 服务和 9102 SSH 隧道。每只手臂只保留一个控制桥，D455 只由一个进程占用。

```bash
# 板子工程根目录：验证 CLI 和真实模型服务
python3 -m cix_harness.diagnose --help
python3 -m cix_harness check-vla

# 真实观测 + 16 步模型预测留档 + 第一步只读 IK 检查
python3 -m cix_harness.diagnose --subtask 'pick up the cyan block'
```

模型输出仍需真实双视角及实测状态。预览桥必须是只读，新协议为 diagnostic_protocol=1。未 commissioning 的 profile 可以尝试只读 inspect，但 IK 可达不证明真实工作区、TCP 或接触几何正确。预览不成功时先查日志，不能改 commissioned 标志绕过。

## 空中单动作执行的前提

本轮不下探、不接触桌面、不抓放物体。现场人员确认空载、初始姿态、通道与急停流程后，另存一份专用 .local.json 配置，不能覆盖既有任务配置。

专用配置要求：commissioning_scope=supervised_cartesian_diagnostic，motion_budget=1，ik_solver=kdl_continuous，实测 diagnostic_origin_m 和 diagnostic_radius_m（不超过 0.20 m），以及已有 validate_config 要求的有效工作区、坐标旋转、时效和限幅。commissioned=true 只能表示这些诊断条件已经核验，不表示相机外参或全桌面抓取标定完成。没有现场测量值时不要猜填。

保留原有限幅和速度检查。位置到位阈值可达 1 cm，几毫米测试不能只看 target_reached，需比较实际位移。诊断预算一次耗尽；不要为了不断重试而反复重启。

控制电脑通过该专用配置、--arm-motion 启动桥后，先做人工动作检查。以下数值仅为语法示例，不作为现场可执行位移推荐；需先预览并按现场空间决定。

```bash
# 保持夹爪；六个分量是策略坐标系中的米和弧度，不是默认 base 轴
python3 -m cix_harness.diagnose --subtask 'manual axis check' \
  --manual-action 0.005 0 0 0 0 0
```

上面预览时桥仍须只读。进入新的、已核验的现场执行会话后，使用同样入口另加 --execute；每次执行会重新采集观测，不复用预览 token。人工动作不依赖 π0.5 服务。模型动作执行则要求策略 motion_ready，且 policy/bridge 坐标旋转一致；不要只为了放行动作修改模型配置。

```bash
# 只在上述现场前提已满足后：执行最新预测的第一步，夹爪不动
python3 -m cix_harness.diagnose --subtask 'pick up the cyan block' --execute
```

该命令直接执行一次，不在采图后等待人工确认，以免观测过期。确认发生在运行命令前。通信失败或响应丢失时不自动重试；先读真实状态和控制端日志。不要因等待推理而增大时效容差，必要时先只读热机，仍以原始观测时效检查为准。

## 应收集的证据

- 板端 runs/<ID>/：外部和腕部图像、开始时两服务 health、机器人诊断配置、板端源码哈希、完整预测前缀、state_diagnostic、选中动作、IK 结果或执行回执、执行后新图。
- π0.5 服务 trace_file：完整原始动作块、state8、归一化状态、策略配置、检查点版本。路径属于模型服务所在机器，不一定属于控制机或 Windows。
- 控制配置中启用 motion_audit_path（父目录预先存在）：action_transform、dispatch、applied 或 failed。action_transform 记录限幅后的平移/旋转、策略到 base 旋转、投影前位移、前后目标姿态及是否请求桌面投影。单步诊断不做桌面投影。
- telemetry_age_s、gripper_age_s、camera_age_s、wrist_age_s 是主机接收/处理侧年龄诊断，不是双相机硬件同步证据。
- task_success_claimed=false：完成诊断不代表抓取成功。

既有 coordinate_audit 可读取本入口预测与 applied（执行回执）事件；只读 IK 事件不是实测运动。模型推理耗时、限幅和状态越界需结合 trace 与桥日志分析。状态区间标记不是工作区限制，也不证明坐标映射正确。

记录本次测试中谁摆位、哪个手臂、物体位于画面哪侧，以及是否改过任何配置。先区分“策略原始预测错误”“坐标/限幅改变预测”“目标正确但跟踪不符”，再决定代码修改。TCP 和相机外参未核验前，不以本诊断启用接触抓取。
