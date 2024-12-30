from matching.cetralised_platforms.pac.ETC import ETC_pac
from matching.cetralised_platforms.pac.Algorithms import Elimination_Algo_pac, ImproveElimination_Algo_pac, \
    AdaptiveImproveElimination_Algo_pac, NaiveUniformlySampling_pac

pac_algorithms = {
    "ETC": ETC_pac,
    "NaiveUniformlySampling": NaiveUniformlySampling_pac,
    "Elimination": Elimination_Algo_pac,
    "ImprovedElimination": ImproveElimination_Algo_pac,
    "AdapElimination": AdaptiveImproveElimination_Algo_pac
}
