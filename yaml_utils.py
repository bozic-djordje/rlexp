from typing import Dict, Sequence
import yaml, re


_SCI_FLOAT_RE = re.compile(r"""
    ^[+-]?(
        (?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][+-]?[0-9]+)?  # 1.0e-3, .5e4 …
      | [0-9]+[eE][+-]?[0-9]+                                 # 1e-5, 3E2 …
      | \.inf|\.Inf|\.INF|[-+]?\.inf                          # .inf literals
      | \.nan|\.NaN|\.NAN                                     # .nan literals
    )$
""", re.X)

def load_yaml(stream):
    if not getattr(yaml.SafeLoader, "_sci_float_patched", False):
        yaml.SafeLoader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            _SCI_FLOAT_RE,
            list("-+0123456789.")
        )
        yaml.SafeLoader._sci_float_patched = True 
    return yaml.load(stream, Loader=yaml.SafeLoader)

# representer: inline if every element is a scalar
def flow_style_if_simple(dumper: yaml.Dumper, seq: Sequence):
    simple = all(isinstance(x, (str, int, float, bool, type(None))) for x in seq)
    return dumper.represent_sequence("tag:yaml.org,2002:seq",
                                     seq, flow_style=simple)

yaml.SafeDumper.add_representer(list, flow_style_if_simple)
yaml.SafeDumper.add_representer(
    str,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", data, style="'"
    )
)

def save_yaml(data: Dict, path: str, indent:int=2) -> None:
    with open(path, 'w') as file:
        yaml.safe_dump(
            data,
            file,
            default_flow_style=False,
            sort_keys=False,
            indent=indent,
            width=120
        )
