from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.config_store import ConfigSearchPath


class CataractSAM2SearchPathPlugin(SearchPathPlugin):
    """Register CataractSAM2's config directory with Hydra."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="cataractsam2_cfg", path="pkg://cataractsam2/cfg")
