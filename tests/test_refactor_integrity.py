"""Post-refactor structural and runtime integrity test for Skill Library v1.0."""
import unittest
from pathlib import Path
from environments.saycan_grid import SayCanGridEnv
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime
from skills.registry import SkillLibraryV1
from skills.unified_library import SkillLibraryV1 as FacadeSkillLibraryV1

EXPECTED_SKILLS = {"Explore","Find","NavigateTo","Reach","PickUp","OpenDoor","UnlockDoor","PlaceNear","MoveObstacle","Done"}

class RefactorIntegrityTests(unittest.TestCase):
    def make_runtime(self, seed):
        env = SayCanGridEnv(render_mode=None)
        obs, _ = env.reset(seed=seed)
        runtime = ObservationRuntime(env)
        runtime.reset(obs, initial_inventory="empty")
        self.addCleanup(env.close)
        return runtime

    def test_registry_and_facade_expose_same_ten_skills(self):
        runtime = self.make_runtime(0)
        for cls in (SkillLibraryV1, FacadeSkillLibraryV1):
            library = cls(runtime)
            self.assertEqual(set(library.names), EXPECTED_SKILLS)
            self.assertTrue(all(callable(s.step) for s in library.skills.values()))

    def test_unknown_and_invalid_calls_have_uniform_results(self):
        library = SkillLibraryV1(self.make_runtime(1))
        result = library.execute({"name":"NotARealSkill","args":{}})
        self.assertEqual(result["status"], "unknown_skill")
        for key in ("success","status","reason","skill","actions_executed"):
            self.assertIn(key, result)
        result = library.execute({"name":"PickUp","args":{}})
        self.assertEqual(result["status"], "invalid_arguments")

    def test_planner_wires_to_modular_registry(self):
        planner = UnifiedSayCanPlanner(self.make_runtime(2))
        self.assertIsInstance(planner.library, SkillLibraryV1)

    def test_registry_has_no_legacy_skill_imports(self):
        import skills.registry as module
        source = Path(module.__file__).read_text(encoding="utf-8")
        for token in ("skills.approach_door","skills.exploration","skills.explore_for_door","skills.interaction ","skills.manipulation ","skills.movement"):
            self.assertNotIn(token, source)

    def test_one_registry_call_executes_at_most_one_primitive_action(self):
        runtime = self.make_runtime(3)
        before = runtime.action_count
        result = SkillLibraryV1(runtime).execute({"name":"Explore","args":{}})
        delta = runtime.action_count - before
        self.assertLessEqual(delta, 1)
        self.assertEqual(delta, result.get("actions_executed", 0))

if __name__ == "__main__":
    unittest.main()
