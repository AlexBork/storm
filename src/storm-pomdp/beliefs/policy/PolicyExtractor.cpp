#include "PolicyExtractor.h"
namespace storm::pomdp::policy {

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
PolicyExtractor<PomdpModelType, BeliefValueType, BeliefMdpValueType>::PolicyExtractor(PomdpModelType const& pomdp, BeliefMdpType const& beliefMdp,
                                                                                      storm::storage::Scheduler<BeliefMdpValueType> const& beliefMdpScheduler)
    : beliefMdp(beliefMdp), beliefMdpScheduler(beliefMdpScheduler) {
    // Intentionally left empty
}

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
ObservationBasedFiniteStateController<typename PomdpModelType::ValueType> PolicyExtractor<PomdpModelType, BeliefValueType, BeliefMdpValueType>::exportPolicyAsFiniteStateController() const {
    return;
}

}  // namespace storm::pomdp::policy