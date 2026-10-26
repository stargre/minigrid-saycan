import json

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.manipulation import PickSkill, PlaceSkill
from skills.interaction import OpenDoorSkill
from skills.library import SkillLibrary
from utils.visualization import FrameRecorder
from perception.observation_summary import observation_to_belief


def main():
    env = SayCanGridEnv(render_mode="rgb_array")
    env.reset(seed=42)

    recorder = FrameRecorder(env)
    movement = MovementSkill(recorder)
    picker = PickSkill(recorder, movement)
    opener = OpenDoorSkill(recorder, movement)
    placer = PlaceSkill(recorder, movement)

    library = SkillLibrary(
        env, movement, picker, opener, placer
    )

    task = (
        "Pick up the red ball, open the yellow door, "
        "and place it above the green goal."
    )

    # 仅展示从局部观测生成的摘要，不打印隐藏坐标。
    belief = observation_to_belief(
        env.unwrapped.gen_obs(),
        inventory_status=library.inventory_status,
        completed_subgoals=library.completed_subgoals,
    )

    print("=" * 65)
    print("OBSERVATION-BASED INPUT TO QWEN")
    print("=" * 65)
    print(json.dumps(belief, ensure_ascii=False, indent=2))

    # 只评分，不执行任何技能。
    candidates = library.score_candidates(task)

    print("\n" + "=" * 65)
    print("LLM RELEVANCE VS LEGACY FEASIBILITY")
    print("=" * 65)
    print(
        f"{'Skill':<12}"
        f"{'LLM relevance':>16}"
        f"{'Feasibility':>15}"
        f"{'Combined':>12}"
    )

    for item in candidates:
        print(
            f"{item['name']:<12}"
            f"{item['relevance']:>16.3f}"
            f"{item['feasibility']:>15.3f}"
            f"{item['score']:>12.3f}"
        )
        print("  LLM reason:", item["rationale"])

    print("\nAll LLM scores, including explore:")
    print(json.dumps(
        library.last_llm_scores,
        ensure_ascii=False,
        indent=2,
    ))

    print("\nPASS: scoring test completed; no skill was executed.")

    env.close()


if __name__ == "__main__":
    main()
