"""Where a crypto qlib experiment runs: the qlib workspace with the crypto data and settings.

The upstream qlib loop runs its workflow configs in docker (``QTDockerEnv``) or conda
(``QlibCondaEnv``), as ``MODEL_CoSTEER_ENV_TYPE`` says. Both are kept here; only two things change:

* preparing docker does not download the CN data -- the crypto data must already be under
  ``~/.qlib``, which docker mounts at ``/root/.qlib``;
* every run gets the crypto settings (data directory, pool, benchmark, costs, time segments), which
  replace the CN time segments the upstream runners pass.
"""

from pathlib import Path

from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.scenarios.crypto.qlib.conf import CRYPTO_QLIB_SETTINGS, CryptoQlibSettings
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
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


class CryptoQlibFBWorkspace(QlibFBWorkspace):
    def qlib_env(self):
        return crypto_qlib_env()

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs):
        """Run a workflow config with the crypto settings in place of the runner's segments. The end
        of the test segment is always set: the runner would otherwise pass its own (CN) one."""
        env = {**run_env, **CRYPTO_QLIB_SETTINGS.run_env()}
        env["test_end"] = CRYPTO_QLIB_SETTINGS.test_end or last_test_day()
        return super().execute(qlib_config_name, env, *args, **kwargs)


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
