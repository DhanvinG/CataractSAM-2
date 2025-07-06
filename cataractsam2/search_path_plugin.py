from hydra.core.plugins import SearchPathPlugin
from hydra.core.utils import  SearchPath, SearchPathItem

class CataractSAM2SearchPathPlugin(SearchPathPlugin):
    """Hydra plugin to add cataractsam2/cfg to the config search path."""
    def manipulate_search_path(self, search_path: "SearchPath") -> None:
        # provider name can be anything unique
        search_path.append(
            SearchPathItem(provider="cataractsam2_cfg",
                           path="pkg://cataractsam2/cfg")
        )
