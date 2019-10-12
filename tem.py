import yaml

class AttrDict(dict):
    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif item in self:
            return self[item]
        elif item.startswith("__"):
            raise ArithmeticError
        else:
            self[item] = AttrDict()

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        elif key in self:
            self[key] = value

    def __str__(self):
        return yaml.dump(self.strip(), default_flow_style=False)

    def strip(self):
