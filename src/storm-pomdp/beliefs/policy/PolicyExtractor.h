#pragma once
#include "ObservationBasedFiniteStateController.h"
#include "storm-pomdp/beliefs/storage/Belief.h"

namespace storm {
namespace models::sparse {
template<class ValueType>
class Mdp;
}
namespace storage {
template<typename ValueType>
class Scheduler;
}
namespace pomdp::policy {
template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
class PolicyExtractor {
    using BeliefType = storm::pomdp::beliefs::Belief<BeliefValueType>;
    using BeliefMdpType = storm::models::sparse::Mdp<BeliefMdpValueType>;

   public:
    PolicyExtractor(PomdpModelType const& pomdp, BeliefMdpType const& beliefMdp, storm::storage::Scheduler<BeliefMdpValueType> const& beliefMdpScheduler);

    ObservationBasedFiniteStateController<typename PomdpModelType::ValueType> exportPolicyAsFiniteStateController() const;

   private:
    BeliefMdpType const& beliefMdp;
    storm::storage::Scheduler<BeliefMdpValueType> const& beliefMdpScheduler;
};
}  // namespace pomdp::policy
}  // namespace storm