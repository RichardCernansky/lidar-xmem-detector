from dataclasses import dataclass
from collections import deque
from typing import Any, Optional, Dict



def _to_float(x: Any) -> float:
    if x is None:
        return 0.0
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "item"):
        x = x.item()
    return float(x)


@dataclass
class Metric:
    w: int
    window: deque
    total: float = 0.0
    n: int = 0

    @classmethod
    def make(cls, w: int) -> "Metric":
        return cls(w=w, window=deque(maxlen=w))

    def push(self, v: float) -> None:
        self.window.append(v)
        self.total += v
        self.n += 1

    def avg(self) -> float:
        return self.total / self.n if self.n > 0 else 0.0

    def win_avg(self) -> Optional[float]:
        return (sum(self.window) / len(self.window)) if len(self.window) > 0 else None


class TrainStats:
    def __init__(self, w: int = 200):
        self.w = w
        self.total = Metric.make(w)
        self.det = Metric.make(w)
        self.occ = Metric.make(w)
        self.cons = Metric.make(w)
        self.cls = Metric.make(w)
        self.loc = Metric.make(w)
        self.dir = Metric.make(w)
        self.aux = Metric.make(w)

    def update_main(self, loss_total_v: float, loss_det_v: float, loss_occ_v: float, loss_cons_v: float) -> None:
        self.total.push(loss_total_v)
        self.det.push(loss_det_v)
        self.occ.push(loss_occ_v)
        self.cons.push(loss_cons_v)

    def update_optional(self, tb_dict: Dict[str, Any]) -> None:
        if "rpn_loss_cls" in tb_dict:
            self.cls.push(_to_float(tb_dict["rpn_loss_cls"]))
        if "rpn_loss_loc" in tb_dict:
            self.loc.push(_to_float(tb_dict["rpn_loss_loc"]))
        if "rpn_loss_dir" in tb_dict:
            self.dir.push(_to_float(tb_dict["rpn_loss_dir"]))
        if "aux_motion_tf" in tb_dict:
            self.aux.push(_to_float(tb_dict["aux_motion_tf"]))


def _extract_losses(tb_dict: Dict[str, Any], loss_tensor: Any) -> tuple[float, float, float, float]:
    loss_total_v = _to_float(tb_dict.get("loss_total", loss_tensor))
    loss_det_v = _to_float(tb_dict.get("loss_det", 0.0))
    loss_occ_v = _to_float(tb_dict.get("loss_aux_occ", 0.0))
    loss_cons_v = _to_float(tb_dict.get("loss_aux_cons", 0.0))
    return loss_total_v, loss_det_v, loss_occ_v, loss_cons_v


def _format_debug(batch_dict: Dict[str, Any]) -> str:
    dbg_s = batch_dict.get("_dbg_s", None)
    if dbg_s is None:
        return ""
    dbg_bev_l1 = batch_dict.get("_dbg_bev_l1", None)
    dbg_temp_l1 = batch_dict.get("_dbg_temp_l1_scaled", None)
    dbg_temp_ratio = batch_dict.get("_dbg_temp_ratio", None)
    dbg_delta_ratio = batch_dict.get("_dbg_delta_ratio", None)
    return (
        f"s {_to_float(dbg_s):.3f}, "
        f"bev_l1 {_to_float(dbg_bev_l1):.3e}, "
        f"temp_l1 {_to_float(dbg_temp_l1):.3e}, "
        f"temp_ratio {_to_float(dbg_temp_ratio):.3f}, "
        f"delta_ratio {_to_float(dbg_delta_ratio):.3f}, "
    )


def log_train_step(
    logger,
    stats: TrainStats,
    epoch: int,
    total_epochs: int,
    seq_idx: int,
    num_seqs: int,
    lr: float,
    loss_total_v: float,
    loss_det_v: float,
    loss_occ_v: float,
    loss_cons_v: float,
    tb_dict: Dict[str, Any],
    batch_dict: Dict[str, Any],
) -> None:
    win_loss = stats.total.win_avg()
    win_det = stats.det.win_avg()
    win_occ = stats.occ.win_avg()
    win_cons = stats.cons.win_avg()

    loss_str = (
        f"epoch {epoch + 1}/{total_epochs}, seq {seq_idx + 1}/{num_seqs}, "
        f"loss {loss_total_v:.4f}, det {loss_det_v:.4f}, occ {loss_occ_v:.4f}, cons {loss_cons_v:.4f}, "
        f"win200 loss {win_loss:.4f}, det {win_det:.4f}, occ {win_occ:.4f}, cons {win_cons:.4f}, "
    )

    if "rpn_loss_cls" in tb_dict:
        win_cls = stats.cls.win_avg()
        loss_str += f"cls {_to_float(tb_dict['rpn_loss_cls']):.4f}, win200_cls {win_cls:.4f}, "
    if "rpn_loss_loc" in tb_dict:
        win_loc = stats.loc.win_avg()
        loss_str += f"loc {_to_float(tb_dict['rpn_loss_loc']):.4f}, win200_loc {win_loc:.4f}, "
    if "rpn_loss_dir" in tb_dict:
        win_dir = stats.dir.win_avg()
        loss_str += f"dir {_to_float(tb_dict['rpn_loss_dir']):.4f}, win200_dir {win_dir:.4f}, "
    if "aux_motion_tf" in tb_dict:
        win_aux = stats.aux.win_avg()
        loss_str += f"aux {_to_float(tb_dict['aux_motion_tf']):.4f}, win200_aux {win_aux:.4f}, "

    loss_str += _format_debug(batch_dict)
    loss_str += f"lr {lr:.6e}"
    logger.info(loss_str)


def log_train_epoch_summary(logger, stats: TrainStats, epoch: int, total_epochs: int) -> None:
    msg = (
        f"epoch {epoch + 1}/{total_epochs} summary: "
        f"avg_total {stats.total.avg():.4f}, "
        f"avg_det {stats.det.avg():.4f}, "
        f"avg_occ {stats.occ.avg():.4f}, "
        f"avg_cons {stats.cons.avg():.4f}"
    )

    if stats.cls.n > 0:
        msg += f", avg_cls {stats.cls.avg():.4f}"
    if stats.loc.n > 0:
        msg += f", avg_loc {stats.loc.avg():.4f}"
    if stats.dir.n > 0:
        msg += f", avg_dir {stats.dir.avg():.4f}"
    if stats.aux.n > 0:
        msg += f", avg_aux {stats.aux.avg():.4f}"

    if len(stats.total.window) > 0:
        msg += f", win200_total {stats.total.win_avg():.4f}"
    if len(stats.det.window) > 0:
        msg += f", win200_det {stats.det.win_avg():.4f}"
    if len(stats.occ.window) > 0:
        msg += f", win200_occ {stats.occ.win_avg():.4f}"
    if len(stats.cons.window) > 0:
        msg += f", win200_cons {stats.cons.win_avg():.4f}"

    logger.info(msg)