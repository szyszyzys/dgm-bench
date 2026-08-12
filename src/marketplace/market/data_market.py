from abc import ABC, abstractmethod
from typing import Dict, Any




class DataMarketplace(ABC):
    @abstractmethod
    def register_seller(self, seller_id: str, seller: Any):
        """Register a seller in the marketplace."""
        pass

