from core.config import TUNABLES


class RuntimeConfig:

    def __init__(self):
        self.values = TUNABLES.copy()

    def set(self, key, value):
        self.values[key] = value

    def get(self, key, default=None):
        return self.values.get(key, default)

    def update(self, d):
        self.values.update(d)

    def reset(self):
        self.values = TUNABLES.copy()

    def as_dict(self):
        return dict(self.values)

    def load_profile(self, module_name):
        """
        Пример:
            runtime_config.load_profile(
                "configs.tuned_stable"
            )
        """

        import importlib

        module = importlib.import_module(
            module_name
        )

        loaded = {}

        for key in dir(module):

            if not key.isupper():
                continue

            value = getattr(module, key)

            self.set(key, value)

            loaded[key] = value

        print(
            f"\n✅ Loaded tuning profile: "
            f"{module_name}"
        )

        for k, v in loaded.items():
            print(f"{k} = {v}")


runtime_config = RuntimeConfig()