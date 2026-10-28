# MiniGrid SayCan 底层技能库 v1.0：中文详细说明

> 文档范围：对应仓库当前 main 分支中的模块化 Skill Library v1.0。本文重点解释底层技能、观测与信念状态、动作执行边界、技能组合、规划器如何调用技能，以及当前验证结果和能力边界。它是面向开发者和研究者的代码导读，不代表系统已经具备真实机械臂控制能力。

## 1. 先用一句话理解这套系统

这套系统把自然语言任务转化为一组可复用、带参数的语义技能，再由确定性的控制器逐步执行 MiniGrid 原子动作；每次动作之后，系统只利用新观测和已执行动作更新自己的信念状态，并据此检查进度、选择下一项技能。

它不是让 LLM 直接输出一长串左转、右转、前进、拾取等原子动作。LLM（或离线关键词评分器）只参与“当前哪项语义技能与任务更相关”的评分；技能是否可执行由确定性的可行性模块判断；真正改变环境的动作只能经过统一的运行时网关。

## 2. 必须区分的三个层次

### 2.1 原子动作（primitive action）

原子动作是 MiniGrid 环境真正执行的一步，例如向前移动、左转、右转、拾取、放下、切换门状态或发出 Done 动作。它们由 MiniGrid 的 Actions 定义。

原子动作回答的是：“这一刻具体做什么动作？”

### 2.2 语义技能（semantic skill）

语义技能是有明确目标、参数、前置条件和结果状态的控制器，例如：

- Find(target)：寻找满足目标描述的对象。
- NavigateTo(target)：走到目标旁边并面向目标。
- PickUp(target)：拾取指定物体并确认持有状态。
- OpenDoor(door)：打开已观察到的、未上锁的关闭门。
- PlaceNear(object, target, relation)：把手持物放到目标指定的相对位置。
- MoveObstacle(object, destination, relation)：组合拾取与放置来移动障碍物。

技能回答的是：“为了完成某个子目标，控制器应该如何根据当前观测逐步行动？”

一个技能可能需要多次调用才能完成，但**每次注册表调用最多执行一个原子动作**。如果当前已经满足目标或前置条件不成立，调用也可能不执行任何动作。

### 2.3 高层规划（planner）

规划器根据完整任务文本、候选技能、相关性评分、可行性评分和当前信念状态，决定下一项要执行的技能。它负责选择、重复调用、超时控制、失败处理、记录历史和任务级完成判断。

规划器回答的是：“在当前状态下，下一步最值得尝试的语义技能是什么？”

这三层分离后，同一个 Find 技能可以查找不同颜色、不同类型的对象；复杂任务通过技能参数化和组合实现，而不需要为每个任务或每种物体复制一套控制器。

## 3. 总体架构与一次任务的执行路径

整体数据流如下：

自然语言任务
→ TaskDependencyResolver 生成技能候选和前置技能
→ SkillFeasibilityEvaluator 检查候选是否可执行
→ RelevanceScorer 评估候选与任务的语义相关性
→ UnifiedSayCanPlanner 依据相关性 × 可行性选择技能
→ SkillLibraryV1 注册表分发技能调用
→ 具体技能控制器根据 BeliefState 决定是否行动
→ ObservationRuntime.execute_action 执行一个 MiniGrid 原子动作
→ 获取新观测、更新 BeliefState
→ 技能继续执行至完成、失败或预算耗尽
→ 规划器记录结果并重新评估候选
→ 依据信念状态检查任务是否完成

关键边界：

1. 目标描述和参数在技能调用之间传递，不是硬编码的对象坐标。
2. 感知与信念状态提供“智能体目前知道什么”，不等于环境的完整真实状态。
3. 相关性和可行性是两个独立维度。任务中提到某项技能，不意味着该技能现在一定能执行。
4. 具体技能可以使用其他技能控制器组合复杂行为，但不能绕过运行时网关直接调用环境。
5. 测试和评估代码可以在执行后检查真实环境状态；真实状态不能作为技能决策输入。

## 4. 代码目录：每个关键文件负责什么

### 4.1 信念状态与观测处理

- **perception/belief_state.py**：核心 BeliefState。保存智能体估计位置、方向、已知地图、手持物、库存状态、已完成子目标和观测计数；用前后两次观测及刚刚执行的动作推断状态变化。
- **perception/observation_summary.py**：把一帧局部观测转换为结构化摘要，例如可见物体、颜色、相对方位和信息限制。它不提供未观测对象的全局坐标。
- **environments/saycan_grid.py**：用于自定义 SayCan 任务的环境适配与回归测试。它不应成为技能读取隐藏地图的入口。

### 4.2 底层控制与技能

- **skills/observation_runtime.py**：唯一允许执行 MiniGrid 原子动作的运行时网关；调用 env.step，更新动作计数、奖励、终止标记和 BeliefState，并刻意丢弃环境返回的 info。
- **skills/navigation.py**：ObservationNavigator。只根据已知地图寻找路径，并以单步方式转向或前进；还负责走到目标相邻格并面向目标。
- **skills/observation_explorer.py**：ObservationExplorer。寻找已知可通行格与未知格之间的探索前沿，移动至前沿并转向未知区域，借助后续观测扩展地图。
- **skills/target.py**：统一目标描述、对象别名、颜色/状态匹配和空间关系解析。
- **skills/base.py**：UnifiedSkill 基类，提供共享运行时、信念状态、导航器、目标位置查询和统一结果结构。
- **skills/search.py**：实现 Explore 与 Find。
- **skills/navigation_skills.py**：实现 NavigateTo 与 Reach。
- **skills/doors.py**：实现 OpenDoor 与 UnlockDoor。
- **skills/manipulation_skills.py**：实现 PickUp、PlaceNear 和复合技能 MoveObstacle。
- **skills/interaction_skill.py**：实现 Done。
- **skills/registry.py**：定义 SkillLibraryV1 的正式技能注册表和统一 execute(call) 接口。
- **skills/unified_library.py**：兼容旧导入路径的门面（facade），把调用转发到模块化实现，不再集中存放技能逻辑。
- **skills/library.py**：旧调用方兼容适配器。新代码优先使用 SkillLibraryV1。

### 4.3 高层规划与技能选择

- **skills/dependency.py**：TaskDependencyResolver。将任务中的对象和动作意图转换为带参数的技能候选，并生成候选技能的前置条件；它只描述依赖，不执行动作。
- **skills/feasibility.py**：SkillFeasibilityEvaluator。基于当前 BeliefState 检查目标是否已知、路径是否可达、手中是否持有正确物体、门是否处于适当状态、放置格是否已知且为空等。
- **llm/skill_relevance.py**：提供离线关键词相关性评分器和可选的 Qwen 相关性评分器。评分器只负责语义相关性，不负责判断物理可行性。
- **planners/unified_saycan_planner.py**：UnifiedSayCanPlanner。负责候选池、相关性与可行性合并评分、技能执行预算、重试/阻塞记录、执行轨迹、历史记录和任务级完成检查。
- **planners/saycan_planner.py**、**planners/observation_task_controller.py**：为旧调用方保留的兼容入口或独立控制器；具体实验应明确使用哪个入口，避免把不同规划流程的结果混为一谈。

### 4.4 验证入口

- **tests/**：快速单元测试，覆盖注册表/接口契约、目标解析、依赖候选、任务完成证据等。
- **scripts/19_skill_library_v1_smoke_test.py**：检查技能库可导入、注册表可调用，并展示一次 Find 的中间结果。
- **scripts/20_skill_acceptance_suite.py**：跨随机种子验证目标存在时能找到、目标不存在时能明确结束搜索。
- **scripts/21_skill_behavior_suite.py**：分别验证十项公开技能及其关键行为。
- **scripts/22_hidden_state_audit.py**：对决策模块进行 AST 静态审计，并运行受保护的端到端场景，检查隐藏状态访问与动作网关边界。
- **scripts/23_official_doorkey_acceptance.py**：运行官方 MiniGrid DoorKey 5x5 和 6x6 验收场景。
- **scripts/08_saycan_planner_test.py**：运行一个完整的自定义 SayCan 任务，支持通过 SEED 指定随机种子。
- **scripts/09_batch_evaluation.py**：运行五种任务语言表述，并输出 CSV/JSON 指标。

## 5. BeliefState：技能究竟依据什么判断

BeliefState 是技能决策的主要状态来源。它不是完整环境状态的副本，而是智能体根据自身观测和动作历史构建的局部地图与状态估计。

### 5.1 核心字段

- **agent_pos**：智能体估计位置。重置时选择任意原点；坐标是内部相对坐标，不是 MiniGrid 的真实全局坐标。
- **agent_dir**：估计朝向。代码使用东、南、西、北四个方向，分别对应方向索引 0、1、2、3。
- **known_map**：已观察到的格子映射。格子记录对象类型、颜色、状态及最近一次观测计数。未知格子不应被当作空地。
- **inventory_status**：库存状态，例如 empty 或某种颜色与物体组合；初始值由 reset 时的设置给出。
- **held_object**：根据拾取前后的观测推断出的手持物信息。
- **completed_subgoals**：通过可观察到的状态变化确认的子目标，例如 pick、pickup、open_door、unlock_door、drop 和 place。
- **observation_count**：已整合的观测次数，可用于判断失败技能之后是否出现了新的信息。

### 5.2 地图如何从局部观测构建

MiniGrid 的观测图像按 image[x, y] 编码。BeliefState 根据当前观测的朝向，把局部视野中的前方/侧向偏移映射到内部相对坐标，再把看到的格子写入 known_map。未知区域保持未知；不能因为某个格子没有出现在当前视野中，就认定它是空地。

### 5.3 位置与朝向如何更新

执行 forward 前，BeliefState 检查前一次观测中正前方的格子是否可通行。只有满足移动条件时，才将估计位置向当前方向推进一格。朝向则由新观测中的 direction 更新。

这意味着位置估计来自“之前看到的前方格子 + 已执行动作 + 新观测”，而不是读取环境内部 agent_pos。

### 5.4 拾取、开门和放置如何确认

- **拾取**：执行 pickup 后，如果此前正前方是可拾取物、库存此前为空，并且新观测显示该物体已从原格消失，BeliefState 才记录手持物和 pick/pickup 子目标。
- **开门**：执行 toggle 后，对比门操作前后的可观察状态。closed → open 记录 open_door；locked → closed/open 记录 unlock_door；若直接变为 open，也记录 open_door。
- **放置**：执行 drop 后，只有新观测确认物体出现在正前方且与原手持物匹配，才清空手持物并记录 drop。
- **特定放置目标**：当前代码还会检查绿色 goal 上方是否观察到红球，以记录自定义任务中的 place 子目标。

因此，“发出过拾取动作”不等于“已经拾取成功”；“发出过开门动作”不等于“门已打开”；技能完成必须依赖更新后的观测证据。

## 6. 唯一动作入口：ObservationRuntime

ObservationRuntime 的 execute_action 是系统唯一允许执行原子动作的入口。它的主要流程是：

1. 确认环境已经 reset，且 episode 尚未结束。
2. 保存上一次观测。
3. 调用环境 step 执行一个动作。
4. 更新 action_count、last_reward、terminated 和 truncated。
5. 将 previous_obs、action、next_obs 交给 BeliefState.update。
6. 保存新观测，并向上层返回动作编号、奖励、终止信息和动作计数。

环境 step 返回的 info 被刻意丢弃，不向技能或规划器开放。技能代码不应通过 runtime.env 读取 env.grid、真实智能体坐标、完整地图或其他隐藏字段。

需要注意：这个边界是代码结构与审计规则共同建立的约束。静态审计和受保护运行测试能够发现已覆盖范围内的违规访问，但不能在数学意义上证明任何未来新增代码都绝不可能泄漏状态。因此，新增决策模块时必须继续更新并运行隐藏状态审计。

## 7. 目标描述：一个技能适配不同物体

### 7.1 TargetSpec

TargetSpec 是统一的目标描述结构，支持对象类型、颜色、状态和可读标签。目标可来自自然语言字符串、字典或已经构建的 TargetSpec。

例如：

    {"object": "ball", "color": "red", "label": "red ball"}

表示红球。技能用 TargetSpec.matches 对已知格子或手持物进行匹配，而不是为每个具体位置写一个专门技能。

### 7.2 当前支持的解析范围

当前解析器通过规则和正则表达式识别常见颜色、对象与门状态。对象别名包括 balls → ball、spheres → ball、keys → key、doors → door、crates → box 等。颜色包括 red、blue、green、purple、yellow、grey/gray；门状态包括 locked、closed、open。

这不是通用语言理解器，也不是视觉模型。它只支持代码明确列出的词汇和规则。复杂指代、长句语义消歧、任意开放词汇对象等，不能假定已被完整支持。

### 7.3 位置证据的来源

目标描述本身不携带真实坐标。只有当目标已出现在 known_map 中时，技能才会从信念地图中得到候选位置。目标还未被观察到时，应先 Find 或 Explore，而不是凭任务文本猜一个位置。

## 8. 导航：只在已知地图上走

导航由 ObservationNavigator 实现，语义层由 NavigateTo 和 Reach 包装。

### 8.1 已知路径搜索

ObservationNavigator 在已知且可通行的格子上运行 BFS，寻找从当前估计位置到目标或接近位置的路径。当前可通行判定包括 empty、floor、goal，以及状态为 open 的 door。

但普通中转路径会排除 goal 格，避免把目标格当作普通道路穿过后破坏目标证据。Reach 可以显式允许指定的 goal 格作为路径终点，但不会把它当作中间节点。

如果目标格未观察到，返回 target_not_known；如果目标已知但当前已知地图中没有路径，返回 no_known_path。系统不会通过读取隐藏地图来“补出”一条路线。

### 8.2 一次只走一步

BFS 只用来确定当前应该朝向哪个下一格，不会把整条路径一次性执行完。如果下一格方向不同，控制器先转向；如果方向已经正确，则向前移动一步。下一次调用会根据更新后的 BeliefState 重新判断。

180 度转向需要两次右转，因此也会拆分成多个原子动作。

### 8.3 NavigateTo 与 Reach 的区别

- **NavigateTo(target)**：目标通常是门、球或其他物体。它的完成条件是站在目标的一个可达相邻格，并面向目标，便于随后交互；不会走进物体格。
- **Reach(target)**：用于进入一个已知且可通行的目标格，例如绿色 goal。目标必须已经观察到并且可通行。成功后，BeliefState 记录 reach 子目标。

因此，想打开门或拾取球时一般使用 NavigateTo；任务要求进入目标格时才使用 Reach。

## 9. 探索：从已知边界向未知区域扩展

ObservationExplorer 采用 frontier exploration 思路。探索前沿是一个已知可通行格，它至少有一个相邻格尚未进入 known_map。

大致过程：

1. 在 known_map 中寻找可通行格。
2. 检查它们相邻的未知格，形成候选前沿。
3. 仅保留当前已知地图中可以通过 BFS 到达的前沿。
4. 按路径长度、距离、坐标和方向等稳定规则选择前沿，减少无意义随机游走。
5. 如果智能体不在前沿位置，就沿已知路径逐步移动过去。
6. 到达前沿后转向未知方向，利用新观测尝试揭示更多格子。
7. 如果转向后仍无法观察到目标未知格，就把这条有向边记录为 blocked edge，避免无限重复尝试。
8. 没有可达前沿时返回 exploration_exhausted；尝试超过上限时返回 exploration_stalled。

探索控制器有界运行，不会在一次调用中无限循环。它不假设未知格一定可通行，也不会把未知区域当作已经看过的空地。

## 10. 十项公开技能：输入、行为与完成语义

注册表在 skills/registry.py 中统一暴露以下十项技能。

### 10.1 Explore

**目的：** 扩展当前已知地图。

**输入：** 不需要特定目标对象。

**行为：** 调用 ObservationExplorer，选择一个可达探索前沿，必要时移动或转向一个原子动作。

**常见结果：**
- action_executed：已经执行一个探索动作。
- exploration_exhausted：没有可达的未知前沿。
- exploration_stalled：在限定尝试次数内没有找到有效前沿。

Explore 的单次调用不是“探索完整张地图”，而是推进一次观测驱动的探索过程。

### 10.2 Find(target)

**目的：** 找到符合参数化描述的对象，并提供观测证据。

**输入：** 目标描述，例如红球、蓝球、黄色门或钥匙。

**行为：**
1. 先检查 known_map 中是否已有匹配目标。
2. 如果目标已知，立即返回 target_found，并给出 belief 中的位置与格子证据，不必执行动作。
3. 如果目标未知，执行一次探索控制器动作，返回 searching。
4. 如果探索前沿耗尽仍未发现目标，返回 target_not_found。

**关键点：** Find 不是一个单独的物体识别网络。它使用已有观测证据，并在证据不足时通过探索获取新观测。目标不存在或当前无法继续搜索时，它必须明确失败，不能无证据地声称找到。

### 10.3 NavigateTo(target)

**目的：** 接近一个已经观察到的目标，并面向它。

**前置条件：** 目标应当在 known_map 中，并且至少存在一条到达目标相邻可通行格的已知路径。

**行为：** 选择目标位置，调用 ObservationNavigator.step_to_adjacent。每次调用至多移动或转向一步；到达相邻格并面向目标后返回 skill_complete。

**典型用途：** 拾取物体前接近物体，或者开门前站到门前。

### 10.4 Reach(target)

**目的：** 进入目标所在的可通行格。

**前置条件：** 目标已知且可通行。

**行为：** 记录当前目标，使用允许目标格作为终点的 BFS 逐步接近；当 BeliefState 中的估计位置到达目标格时，记录 reach 并返回 skill_complete。

**与 NavigateTo 的区别：** NavigateTo 停在目标旁边；Reach 要进入目标格。对门等不可通行目标调用 Reach 会被拒绝。

### 10.5 PickUp(target)

**目的：** 拾取指定的可移动对象。

**前置条件：**
- 目标已被观察到；
- 当前库存被确认为空，或者目标已经在手中；
- 目标属于 MiniGrid 支持的可拾取对象（ball、key、box）。

**行为：**
1. 如果匹配物体已经在手中，直接报告完成。
2. 如果库存不为空且不是目标物体，返回 gripper_not_empty。
3. 如果目标未知，返回 target_not_found。
4. 如果目标类型不可拾取，返回 not_pickable。
5. 否则先逐步导航到目标相邻格并面向它，再执行 pickup。
6. 读取更新后的 BeliefState；只有持有物与目标匹配才返回 skill_complete，否则返回 pickup_not_confirmed。

### 10.6 OpenDoor(target)

**目的：** 打开已经观察到的关闭且未上锁的门。

**前置条件：** 目标门已知，状态为 closed，并且智能体能够接近门前。

**行为：**
- 已经观察到门 open：直接返回 skill_complete。
- 门为 locked：返回 requires_key，提示先取得匹配钥匙并调用 UnlockDoor。
- 门状态不明或不是 closed：返回 unknown_door_state。
- 门为 closed：逐步接近并面向门，之后执行 toggle；只有新观测确认门 open，才认为打开成功。

**完成证据：** 新观测中的门状态，而不是仅仅执行了 toggle。

### 10.7 UnlockDoor(target)

**目的：** 使用匹配的钥匙解锁已上锁的门。

**前置条件：**
- 门已知且状态为 locked；
- 手持物必须是颜色匹配的 key；
- 智能体能够接近门前。

**行为：** 检查手持钥匙与门的颜色是否匹配，逐步导航到门前，再执行 toggle。MiniGrid 的锁交互在某些场景下可能直接从 locked 变为 open；因此当前实现把更新观测中门变为 closed 或 open 都视为解锁交互已确认。如果变为 open，还会记录 open_door。

没有匹配钥匙时返回 requires_key；门本来就不是 locked 时返回 not_locked。

### 10.8 PlaceNear(object, target, relation)

**目的：** 将当前手持物放在目标的指定相对位置。

**前置条件：**
- 手中持有指定物体；
- 目标对象已被观察到；
- 目标关系对应的放置格已被观察到，并且被认为是空格或地板；
- 智能体可以到达放置格相邻位置。

**当前支持的关系：**
- above / on_top_of：目标格上方一格；
- below：目标格下方一格；
- left_of：目标格左侧一格；
- right_of：目标格右侧一格；
- near / adjacent / nearby / beside / next_to：选择一个已知为空且符合相邻关系的格子。

**行为：**
1. 验证目标物确实在手中。
2. 找到目标的位置。
3. 根据 relation 计算放置格。
4. 检查放置格是否已知且为空。
5. 导航到放置格相邻位置并面向放置格。
6. 执行 drop。
7. 只有更新后的观测确认物体已落在预期格子、手中不再持有它，才返回 skill_complete。

如果放置格未知或被占据，分别返回 placement_cell_unknown 或 placement_cell_occupied；如果物体没有被确认放好，返回 placement_not_confirmed。

**重要限制：** 关系方向使用当前 MiniGrid 内部坐标定义；above 对应 y 减一。它是网格相对关系，不是三维空间中的任意姿态规划。

### 10.9 MoveObstacle(target, destination, relation)

**目的：** 把指定障碍物或可移动物体移动到目标附近或指定相对位置。

**设计方式：** 这是一个组合技能，内部复用 PickUpSkill 与 PlaceNearSkill，而不是重新实现一套拾取和放置逻辑。

**内部阶段：**
1. pickup：调用 PickUp 控制器。接近目标时逐步执行；只有 BeliefState 确认目标在手中，才切换到 place。
2. place：调用 PlaceNear 控制器，执行接近、放置和结果验证。
3. 完成后重置组合技能阶段，便于后续再次使用。

**防错逻辑：** 阶段转换以 BeliefState 中的真实持有证据为准。如果内部阶段记录为 place，但当前信念状态并未确认持有目标，会退回 pickup 阶段。这样可以避免仅凭过期阶段标记跳过拾取。

每次 MoveObstacle 调用依旧最多执行一个原子动作；复杂性来自跨多次调用维护阶段，而不是一次调用内执行一长串动作。

### 10.10 Done(target=None)

**目的：** 发出 MiniGrid 的 Done 原子动作。

**可选目标：** 如果提供 target，先尝试走到目标相邻格并面向目标；如果未提供目标，则直接发出 Done。

**返回语义：** done_issued 表示 Done 动作已发出，不等价于整个任务已被独立证明成功。最终任务成功仍由规划器的信念状态检查以及评估器的环境结果共同判断。

## 11. 技能统一接口与返回值

### 11.1 调用形式

注册表接受统一的技能调用字典。例如：

    {
      "name": "PickUp",
      "args": {
        "target": {
          "object": "ball",
          "color": "red",
          "label": "red ball"
        }
      }
    }

SkillLibraryV1 根据 name 查找对应控制器，然后把 args 传给该技能的 step 方法。调用方不需要知道具体技能类位于哪个文件。

### 11.2 统一结果字段

每个结果至少应提供：

- success：本次技能结果是否成功。
- status：机器可判断的结果状态。
- reason：可读的原因或进度解释。
- skill：技能名称。
- actions_executed：本次注册表调用执行的原子动作数。

根据技能不同，结果还可能包含 target、evidence、target_position、held_object、placement_position、phase、terminated 等附加字段。

### 11.3 为什么不能只看 success

有些控制器在成功执行一个原子动作后返回 success=True、status=action_executed。这表示动作执行成功，不表示高层技能已经完成。例如，Find 可能已经转向一个未知区域，但尚未找到目标。

规划器用 status 区分中间进度与终态。当前进度状态包括 action_executed、searching 和 phase_complete。只有终态成功才会把该技能调用记为完成；终态失败则可能暂时阻塞该候选，等待新观测后再尝试。

因此，调用方应联合检查 success、status 和任务所需证据，而不能只根据一个布尔值判断整体任务是否结束。

## 12. 依赖解析：复杂行为如何由技能组合实现

TaskDependencyResolver 不会直接执行任何动作。它做两件事：生成候选技能，以及告诉规划器候选技能还缺少哪些前置条件。

### 12.1 生成候选

解析器从任务文本中识别常见对象、动作词和空间关系，并结合当前 known_map 中已经观察到的对象补充目标。随后生成带参数的候选，例如 Find(red ball)、PickUp(red ball)、OpenDoor(yellow door)、PlaceNear(red ball, green goal, above)。

这部分是规则驱动的任务解析，不应理解为对任意自然语言的完整语义解析。

### 12.2 前置条件示例

- 目标不在已知地图中：先提出 Find(target)。
- 拾取目标：需要目标已知、目标可拾取且库存条件满足。
- 解锁门：若门被观察为 locked 且没有匹配钥匙，会提出 Find(key) 与 PickUp(key)；必要时还会提出 NavigateTo(door)。
- 放置物体：需要先持有该物体，并知道目标与放置位置；如果目标关系对应的格子尚未观察到，可能提出 Explore。
- 移动障碍物：需要发现障碍物、拾取它、发现目的地，随后调用 PlaceNear。
- 完成指定目标交互：可能需要先 NavigateTo(target)。

这些是候选依赖，不是不可变的固定脚本。规划器每次得到新观测后，仍会重新检查可行性和相关性，所以执行顺序可以随实际状态变化。

## 13. 可行性与相关性：为什么要分开评分

### 13.1 可行性评分

SkillFeasibilityEvaluator 只根据 BeliefState 和已知地图进行确定性检查。它关注的问题包括：

- 目标是否已被观察到；
- 是否存在到目标或其相邻格的已知路径；
- 是否有可探索的未知前沿；
- 手中是否持有匹配的物体；
- 门是否处于请求操作所需要的状态；
- 放置格是否已知、为空且可到达；
- episode 是否已经结束。

不可执行的技能通常得到零可行性，并可能触发前置技能候选的加入。它不应该通过读取真实地图来提高可行性。

### 13.2 相关性评分

相关性评分器回答：“这项技能和用户的任务文本有多相关？”

当前有两种实现：

- **KeywordRelevanceScorer**：离线关键词回退实现，不依赖外部 LLM，适合烟雾测试和可复现的基础回归。
- **QwenSkillRelevanceScorer**：可选的 Qwen 适配器，需要配置 DASHSCOPE_API_KEY；模型收到任务文本、观测派生的状态摘要和候选技能描述，并只返回每个候选的相关性分数与理由。

Qwen 提示明确要求不要评分物理可行性，也不要推断未观察的位置或隐藏状态。

如果运行时没有配置 API Key，默认使用离线关键词评分器。因此，在未启用 Qwen 的运行日志中，不能把结果描述成 LLM 相关性判断的实验结果。

### 13.3 合并评分

规划器对每个候选计算：

    score = relevance × feasibility

其中 relevance 表示任务语义相关性，feasibility 表示当前信念状态下的可执行程度。之后按分数降序选择候选，并用稳定的候选 ID 作为排序辅助条件。

这一设计的直观意义是：一个与任务高度相关但当前根本不可执行的技能，不应该因为语义相关性高就被直接执行；一个当前可执行但与任务无关的技能，也不应该自动胜出。

## 14. 规划器如何控制技能执行

UnifiedSayCanPlanner 的核心循环可以概括为以下步骤：

1. 根据任务文本和当前信念状态生成候选技能。
2. 将可行性评估发现的前置技能加入候选池。
3. 跳过已经完成或在当前观测版本下被阻塞的候选。
4. 对候选进行一次相关性评分，并计算 relevance × feasibility。
5. 选择得分最高且分数大于零的候选。
6. 重复调用该技能控制器，直到它返回终态、失败、停滞或超过预算。
7. 保存技能调用、评分、结果、执行轨迹和信念状态摘要。
8. 根据状态决定是否记录完成、暂时阻塞或允许以后重试。
9. 重新生成/评估候选，直到任务完成、没有可行候选、episode 结束或预算耗尽。

### 14.1 执行预算

当前规划器设置了全局原子动作预算 max_actions（默认 450）和单个技能的控制器调用预算 max_skill_actions（默认 300）；run 还设置了高层规划步数上限。预算的目的是防止控制器无限循环。

这里要区分“高层技能选择次数”和“底层原子动作次数”。一个 Find 可能经过多次控制器调用和转向才找到目标，因此两类计数并不相同。

### 14.2 失败后的重试

如果某项技能失败，规划器会记录它在当时 observation_count 下被阻塞。只要观测没有变化，就不会无意义地立即重复同一个失败候选；新观测到来后，候选可能重新进入评估。

例如，Find(red ball) 可能在当前可达区域中暂时找不到球，但打开门或移动到另一个区域后，地图发生变化，规划器就有机会重新尝试寻找。技能失败不必然意味着整个任务失败。

### 14.3 任务完成检查

规划器在结束时还会运行 belief-side 的任务完成检查。当前检查覆盖明确的拾取、开门/解锁、放置、进入目标格以及显式 Find/Search/Locate 子句。

对显式搜索要求，系统会检查目标是否仍在已知地图中、是否已被持有，或者当前规划历史中是否存在成功的 Find 结果。这样即使物体拾取后从原地图格消失，或智能体经过某格后该格内容被更新，也不会仅因为目标不再“当前可见”就抹掉之前可靠的发现证据。

不过，任务完成解析仍然是基于有限规则和关键词的机制，不是任意复杂任务的通用形式化验证。评估脚本还应检查环境真实终止状态、目标位置或奖励等独立指标。

## 15. 完整示例：拾取红球、开黄门、把红球放到绿目标上方

假设任务文本是：

    Pick up the red ball, open the yellow door, and place it above the green goal.

系统并不预先硬编码唯一的技能顺序，而是根据候选评分、当前观测和前置条件逐步选择。可能的执行逻辑如下：

1. **发现目标**：如果红球、黄门或绿目标还不在 known_map 中，相关技能的可行性可能为零，依赖解析器就会提出 Find 或 Explore。
2. **接近黄门**：如果门已观察到且可达，规划器可能选择 NavigateTo(yellow door)。控制器逐步转向/前进，直到站到门前并面向门。
3. **开门**：OpenDoor 检查门状态。如果门是 closed，就在门前执行 toggle，并通过新观测确认 open；如果门是 locked，则返回 requires_key，需要走匹配钥匙的前置流程。
4. **找到红球**：Find(red ball) 首先检查已知地图；如果没找到，就执行一次探索动作并在后续调用中继续。
5. **拾取红球**：PickUp(red ball) 检查库存和目标证据，走到球旁边后执行 pickup，并通过更新后的 BeliefState 确认球已在手中。
6. **确定放置位置**：PlaceNear(red ball, green goal, above) 根据绿目标位置和 above 的关系，计算目标格上方一格。该格必须已被观察并且被认为是空的；若未知，需要先探索获取证据。
7. **放置**：控制器走到放置格相邻位置并面向放置格，然后执行 drop。
8. **验证放置**：只有新观测确认红球落到预期位置且库存为空，PlaceNear 才报告完成。
9. **检查整个任务**：规划器根据已完成子目标、持有状态和任务中的显式要求进行任务级检查；测试评估器还可以独立检查真实环境状态。

这是一个由参数化技能和依赖关系构成的闭环，而不是让 LLM 直接决定每一个底层动作。实际运行中，技能顺序可以不同；例如如果红球已经在旁边，拾取可能先于开门发生。

## 16. 官方 DoorKey 任务中的组合逻辑

DoorKey 的核心依赖通常是：发现匹配钥匙 → 拾取钥匙 → 接近上锁的门 → 解锁门（某些场景中一次 toggle 就直接开门）→ 通过门 → 到达目标格。

系统中没有为每个 DoorKey 随机种子单独编写一套技能序列。Find、PickUp、NavigateTo、UnlockDoor、OpenDoor 和 Reach 使用统一接口与参数，根据当前观测和可行性组合。

要注意，官方 DoorKey 验收结果只说明已测试的两个环境 ID 和对应随机种子通过；它不自动证明所有 MiniGrid 环境家族、所有地图规模或所有任务指令都得到支持。

## 17. 当前验证结果与它们能说明什么

根据当前项目记录的服务器测试输出：

- 单元测试：16/16 通过。
- 技能库烟雾测试：通过。
- 搜索验收：20/20 个随机种子通过，目标存在时找到、目标不存在时明确结束。
- 技能行为套件：11/11 组通过，覆盖十项公开技能及统一 API 相关检查。
- 隐藏状态审计：静态扫描 17 个决策模块；受保护端到端种子 0、7、42、45、46 通过。
- 官方 DoorKey：5x5 与 6x6 两种环境各测试五个种子，共 10/10 通过。
- 完整自定义任务：通过。
- 五种任务语言变体：5/5 通过。
- 重构后完整任务回归：20/20 种子通过。
- git status --short 与 git diff --check：无输出。

这些结果支持“当前实现通过了已定义的接口、行为、隐藏状态边界和特定环境回归测试”。它们不等于证明所有未测试任务都成功，也不等于真实机器人上的安全性或鲁棒性验证。

## 18. 当前能力边界：哪些事情还不能默认认为已经实现

1. **目标解析是规则驱动的。** 当前颜色、对象、门状态和空间关系来自明确的词表与规则，不是开放词汇视觉理解。
2. **地图是部分观测地图。** 未知格不能当作空地；没有可达前沿时，Find 会明确失败，而不是保证找到全局任何位置的对象。
3. **导航使用已知格上的 BFS。** 这是离散网格路径规划，不是连续空间运动规划，也没有真实机械臂的动力学、碰撞模型或关节约束。
4. **放置关系有限。** 当前实现支持网格中的上、下、左、右和近邻关系，不支持任意三维姿态、堆叠稳定性或复杂接触规划。
5. **技能执行是闭环的，但规划语言理解并不通用。** 候选生成和任务完成检查仍依赖关键词/正则规则。
6. **相关性模型与物理控制分离。** 默认离线评分器不代表 LLM 规划性能；使用 Qwen 时也必须明确记录模型、配置、API Key 状态和实验设置。
7. **测试覆盖范围有限。** 官方 DoorKey 10 个测试回合、自定义任务五个语言变体和 20 个随机种子是已验证范围，不应外推成对所有环境的保证。
8. **当前是 MiniGrid 技能库。** 接入 Isaac Sim 或真实机械臂时，需要新的环境适配、动作映射、状态估计、控制器与安全验证；不能直接把 MiniGrid 的离散动作当作真实机器人动作。

## 19. 冻结后如何修改

v1.0 的核心契约应当保持稳定：

- 十个公开技能的名称和参数语义保持兼容。
- 注册表每次调用最多执行一个原子动作。
- 所有原子动作继续经过 ObservationRuntime.execute_action。
- 技能决策继续只依赖观测派生的 BeliefState。
- 相关性评分与确定性可行性检查保持分离。
- 中间进度状态与终态成功/失败保持可区分。
- 新增或移动决策模块时同步更新 scripts/22_hidden_state_audit.py。
- 修改后重新运行单元测试、技能行为测试、隐藏状态审计、20 种子回归、批量任务和官方 DoorKey 验收。

如果只是增加新任务指令，应优先扩展候选生成和目标解析，并通过回归测试确认原有技能契约没有改变。如果改变技能本身的状态语义或动作执行边界，则应视为新的版本变更，而不是普通文档修改。

## 20. 建议的阅读顺序

如果要真正掌握代码，建议按下面顺序阅读，而不是从规划器开始：

1. **perception/belief_state.py**：先弄清楚系统“知道什么”，以及位置、物体和子目标证据如何更新。
2. **skills/observation_runtime.py**：弄清楚原子动作怎样执行，环境 info 为什么被丢弃。
3. **skills/target.py**：掌握目标如何被解析、规范化和匹配。
4. **skills/navigation.py** 与 **skills/observation_explorer.py**：掌握底层单步导航与探索算法。
5. **skills/base.py**：理解技能共享的状态与返回结构。
6. **skills/search.py**、**skills/navigation_skills.py**、**skills/doors.py**、**skills/manipulation_skills.py**、**skills/interaction_skill.py**：逐个理解十项公开技能。
7. **skills/registry.py**：理解统一调用接口。
8. **skills/dependency.py** 与 **skills/feasibility.py**：理解技能候选和前置条件。
9. **llm/skill_relevance.py**：理解离线/Qwen 相关性评分的区别。
10. **planners/unified_saycan_planner.py**：最后理解候选评分、技能执行循环、重试和任务完成判断。
11. **scripts/21_skill_behavior_suite.py** 与 **scripts/22_hidden_state_audit.py**：把每个设计约束对应到实际测试。

掌握以上路径后，应能回答三个关键问题：技能为什么现在可以执行？技能做了一步之后，系统从哪里知道发生了什么？什么证据足以让系统认为一个子目标或整个任务已经完成？
