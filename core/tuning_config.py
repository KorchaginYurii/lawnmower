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

    def as_dict(self):
        return dict(self.values)


runtime_config = RuntimeConfig()