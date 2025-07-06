from hydra.core.plugins import SearchPathPlugin
from hydra.core.utils import SearchPath, SearchPathItem

class CataractSAM2SearchPathPlugin(SearchPathPlugin):
    """Hydra plugin that registers the package's default configuration."""

    def manipulate_search_path(self, search_path: SearchPath) -> None:
        """Append CataractSAM2's config directory to Hydra's search paths."""
        search_path.append(
            SearchPathItem(
                provider="cataractsam2_cfg",
                path="pkg://cataractsam2/cfg",
            )
        )

