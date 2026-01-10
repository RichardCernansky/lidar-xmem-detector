from typing import Any, Dict, List, Tuple


class MissingConfigError(RuntimeError):
    pass


def cfg_req_key(root: Any, path: str) -> Any:
    cur = root
    for k in path.split("."):
        if isinstance(cur, dict):
            if k not in cur:
                raise MissingConfigError(f"Missing config key: {path}")
            cur = cur[k]
        else:
            if not hasattr(cur, k):
                raise MissingConfigError(f"Missing config key: {path}")
            cur = getattr(cur, k)
    return cur


def cfg_req_nn(root: Any, path: str) -> Any:
    v = cfg_req_key(root, path)
    if v is None:
        raise MissingConfigError(f"Config key is null but required: {path}")
    return v


def cfg_req_int(root: Any, path: str) -> int:
    v = cfg_req_nn(root, path)
    try:
        return int(v)
    except Exception:
        raise MissingConfigError(f"Config key must be int: {path}")


def cfg_req_float(root: Any, path: str) -> float:
    v = cfg_req_nn(root, path)
    try:
        return float(v)
    except Exception:
        raise MissingConfigError(f"Config key must be float: {path}")


def cfg_req_bool(root: Any, path: str) -> bool:
    v = cfg_req_nn(root, path)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise MissingConfigError(f"Config key must be bool: {path}")


def cfg_req_str(root: Any, path: str) -> str:
    v = cfg_req_nn(root, path)
    if not isinstance(v, str) or len(v.strip()) == 0:
        raise MissingConfigError(f"Config key must be non-empty string: {path}")
    return v


def cfg_req_list_str(root: Any, path: str) -> List[str]:
    v = cfg_req_nn(root, path)
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        raise MissingConfigError(f"Config key must be non-empty list: {path}")
    out: List[str] = []
    for x in v:
        if not isinstance(x, str) or len(x.strip()) == 0:
            raise MissingConfigError(f"Config key must be list of non-empty strings: {path}")
        out.append(x)
    return out


def _require_prob_01(name: str, v: float) -> float:
    if not (0.0 <= float(v) <= 1.0):
        raise MissingConfigError(f"{name} must be in [0,1], got {v}")
    return float(v)


def _require_pos_int(name: str, v: int) -> int:
    if int(v) <= 0:
        raise MissingConfigError(f"{name} must be > 0, got {v}")
    return int(v)
