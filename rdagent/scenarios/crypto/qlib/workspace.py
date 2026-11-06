"""Where a crypto qlib experiment runs: a workspace of its own on the crypto data and settings.

It holds a workflow config template folder and runs one of its configs with ``qrun``, then reads the
results with ``read_exp_res.py``, as the qlib workspace does; the qlib runners call ``execute`` the
same way. The environment is docker (``QTDockerEnv``) or conda (``QlibCondaEnv``), as
``MODEL_CoSTEER_ENV_TYPE`` says, with two differences from the qlib workspace:

* preparing docker does not download the CN data -- the crypto data must already be under
  ``~/.qlib``, which docker mounts at ``/root/.qlib``;
* every run gets the crypto settings (data directory, pool, benchmark, costs, time segments), which
  replace the CN time segments the qlib runners pass.
"""

import re
from pathlib import Path

import pandas as pd

from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.crypto.qlib.conf import CRYPTO_QLIB_SETTINGS, CryptoQlibSettings
from rdagent.utils.env import DockerEnv, QlibCondaConf, QlibCondaEnv, QTDockerEnv

HERE = Path(__file__).parent
FACTOR_TEMPLATE = HERE / "factor_template"
MODEL_TEMPLATE = HERE / "model_template"
FACTOR_DATA_TEMPLATE = HERE / "factor_data_template"


class CryptoDataMissing(RuntimeError):
    """The crypto qlib data directory has not been written."""


def require_data(settings: CryptoQlibSettings = CRYPTO_QLIB_SETTINGS, docker: bool = False) -> Path:
    """The data directory, after checking that it holds a calendar and the pool, and, for docker,
    that it is under ``~/.qlib``: the one directory docker mounts (at ``/root/.qlib``)."""
    directory = settings.data_dir
    mounted = Path("~/.qlib").expanduser()
    if docker and not directory.resolve().is_relative_to(mounted.resolve()):
        raise CryptoDataMissing(
            f"CRYPTO_QLIB_PROVIDER_URI {settings.provider_uri} must be under ~/.qlib to be seen in "
            "docker, which mounts that directory only."
        )
    pool = directory / "instruments" / f"{settings.market}.txt"
    if not (directory / "calendars" / "day.txt").is_file() or not pool.is_file():
        raise CryptoDataMissing(
            f"No crypto qlib data with the pool {settings.market!r} in {directory}. Write it with "
            f"`jarvis-history qlib --out {settings.provider_uri} --top "
            f"{settings.market.removeprefix('top')}` (jarvis-trader-service/scripts), or set "
            "CRYPTO_QLIB_PROVIDER_URI / CRYPTO_QLIB_MARKET."
        )
    return directory


class CryptoQTDockerEnv(QTDockerEnv):
    """The qlib docker image, without the download of the CN data."""

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        DockerEnv.prepare(self, *args, **kwargs)
        require_data(docker=True)


def crypto_qlib_env():
    """The environment `MODEL_COSTEER_SETTINGS.env_type` names; None for an unknown one."""
    if MODEL_COSTEER_SETTINGS.env_type == "docker":
        return CryptoQTDockerEnv()
    if MODEL_COSTEER_SETTINGS.env_type == "conda":
        require_data()
        return QlibCondaEnv(conf=QlibCondaConf())
    return None


def last_test_day(settings: CryptoQlibSettings = CRYPTO_QLIB_SETTINGS) -> str:
    """The last day a backtest may end on: the day before the last of the calendar, since a
    day's trades are settled on the next one."""
    days = (settings.data_dir / "calendars" / "day.txt").read_text().split()
    return days[-2]


TRAINING_LOG = re.compile(r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)")
"""The lines of the qrun log the feedback reads: the losses of each epoch and the best score."""


class CryptoQlibFBWorkspace(FBWorkspace):
    """A workflow config template folder, run on the crypto data with ``qrun``."""

    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(
        self, qlib_config_name: str = "conf.yaml", run_env: dict | None = None, *args, **kwargs
    ) -> tuple[pd.Series | None, str]:
        """Run a workflow config and read its metrics.

        The crypto settings replace the segments in ``run_env`` the runner passes; the end of the test
        segment is always set, since the runner would otherwise pass its own (CN) one. Returns the
        metrics of the latest recorder and the training lines of the log, or None and the whole log
        when the run wrote no results.
        """
        env = {**(run_env or {}), **CRYPTO_QLIB_SETTINGS.run_env()}
        env["test_end"] = CRYPTO_QLIB_SETTINGS.test_end or last_test_day()
        qlib = crypto_qlib_env()
        if qlib is None:
            logger.error(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "Unknown environment type"
        qlib.prepare()

        log = qlib.check_output(local_path=str(self.workspace_path), entry=f"qrun {qlib_config_name}", env=env)
        logger.log_object(log, tag="Qlib_execute_log")
        qlib.check_output(local_path=str(self.workspace_path), entry="python read_exp_res.py", env=env)

        returns = self.workspace_path / "ret.parquet"
        if not returns.exists():
            logger.error("No result file found.")
            return None, log
        logger.log_object(pd.read_parquet(returns), tag="Quantitative Backtesting Chart")

        metrics = self.workspace_path / "qlib_res.csv"
        if not metrics.exists():
            logger.error(f"File {metrics} does not exist.")
            return None, log
        # The run succeeded: keep only the training lines of the log.
        return pd.read_csv(metrics, index_col=0).iloc[:, 0], "\n".join(TRAINING_LOG.findall(log))


__all__ = [
    "CryptoDataMissing",
    "CryptoQTDockerEnv",
    "CryptoQlibFBWorkspace",
    "FACTOR_DATA_TEMPLATE",
    "FACTOR_TEMPLATE",
    "MODEL_TEMPLATE",
    "crypto_qlib_env",
    "last_test_day",
    "require_data",
]
