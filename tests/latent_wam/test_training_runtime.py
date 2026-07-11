import logging
from pathlib import Path

from navsim.planning.script.training_runtime import configure_rank_logging, distributed_rank, pin_experiment_uid


def test_pin_experiment_uid_is_inherited_without_overwrite() -> None:
    environment: dict[str, str] = {}

    assert pin_experiment_uid("parent-uid", environment) == "parent-uid"
    assert environment["NAVSIM_EXPERIMENT_UID"] == "parent-uid"
    assert pin_experiment_uid("child-generated-uid", environment) == "parent-uid"


def test_nonzero_rank_replaces_hydra_file_handler_with_rank_log(tmp_path: Path) -> None:
    root_logger = logging.Logger("training-runtime-test")
    hydra_log = tmp_path / "run_training.log"
    root_logger.addHandler(logging.FileHandler(hydra_log))
    root_logger.addHandler(logging.StreamHandler())

    rank_log = configure_rank_logging(tmp_path, rank=3, root_logger=root_logger)

    assert rank_log == tmp_path / "rank_3.log"
    file_handlers = [handler for handler in root_logger.handlers if isinstance(handler, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].baseFilename == str(rank_log)


def test_distributed_rank_combines_node_and_local_rank() -> None:
    assert distributed_rank({"NODE_RANK": "2", "LOCAL_RANK": "3", "LOCAL_WORLD_SIZE": "8"}) == 19
    assert distributed_rank({"RANK": "7", "LOCAL_RANK": "1"}) == 7


def test_rank_zero_keeps_hydra_logging_unchanged(tmp_path: Path) -> None:
    root_logger = logging.Logger("training-runtime-rank-zero-test")
    handler = logging.FileHandler(tmp_path / "run_training.log")
    root_logger.addHandler(handler)

    assert configure_rank_logging(tmp_path, rank=0, root_logger=root_logger) is None
    assert root_logger.handlers == [handler]

    handler.close()
