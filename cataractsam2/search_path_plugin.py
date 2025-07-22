from hydra.core.plugins import SearchPathPlugin
from hydra.core.utils import SearchPath, SearchPathItem

class CataractSAM2SearchPathPlugin(SearchPathPlugin):

    def manipulate_search_path(self, search_path: SearchPath) -> None:
        search_path.append(
            SearchPathItem(
                provider="cataractsam2_cfg",
                path="pkg://cataractsam2/cfg",
            )
        )

