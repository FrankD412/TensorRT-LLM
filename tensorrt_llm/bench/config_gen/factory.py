from typing import Dict, List

from aenum import MultiValueEnum

from tensorrt_llm.bench.config_gen.recipes import ScenarioProtocol


class BackendType(MultiValueEnum):
    TENSORRT = "TensorRT", "trt", "tensorrt"
    PYTORCH = "PyTorch", "torch", "pytorch"
    AUTODEPLOY = "AutoDeploy", "autodeploy", "_autodeply"

    def __str__(self):
        return self.values[-1]


class RecipeFactory:
    _scenarios: Dict[str, ScenarioProtocol] = {}

    @classmethod
    def register(cls, scenario_name: str):

        def decorator(recipe_cls: ScenarioProtocol):
            if scenario_name in cls._scenarios:
                raise ValueError(f"Scenario {scenario_name} already registered")
            cls._scenarios[scenario_name] = recipe_cls
            return recipe_cls

        return decorator

    @classmethod
    def get_available(cls) -> List[str]:
        return list(cls._scenarios.keys())

    @classmethod
    def get_recipe(cls, scenario_name: str) -> type:
        if scenario_name not in cls._scenarios:
            raise ValueError(
                f"Scenario {scenario_name} not found. Available scenarios: "
                f"{cls.get_available_scenarios()}")
        return cls._scenarios[scenario_name]


class BackendFactory:
    _factories: Dict[FrameworkName, RecipeFactoryInterface] = {}

    @classmethod
    def register_framework(cls, backend: BackendType,
                           factory_cls: RecipeFactoryInterface):
        cls._factories[framework_name] = factory_cls

    @classmethod
    def get_factory(cls,
                    framework_name: FrameworkName) -> RecipeFactoryInterface:
        return cls._factories[framework_name]

    @classmethod
    def list_available_frameworks(cls) -> List[FrameworkName]:
        return list(cls._factories.keys())
