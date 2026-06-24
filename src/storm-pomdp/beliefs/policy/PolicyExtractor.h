#pragma once
#include "ObservationBasedFiniteStateController.h"
#include "storm-pomdp/beliefs/storage/Belief.h"

namespace storm {
namespace models::sparse {
template<typename ValueType>
class StandardRewardModel;
template<typename ValueType, typename RewardModelType>
class Mdp;
template<typename ValueType, typename RewardModelType>
class Model;
}  // namespace models::sparse
namespace storage {
template<typename ValueType>
class Scheduler;
}
namespace pomdp::policy {
template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
class PolicyExtractor {
    using BeliefType = storm::pomdp::beliefs::Belief<BeliefValueType>;
    using BeliefMdpType = storm::models::sparse::Mdp<BeliefMdpValueType, storm::models::sparse::StandardRewardModel<BeliefMdpValueType>>;
    using ModelType = storm::models::sparse::Model<BeliefMdpValueType, storm::models::sparse::StandardRewardModel<BeliefMdpValueType>>;

   public:
    PolicyExtractor(
        PomdpModelType const& pomdp, BeliefMdpType const& beliefMdp, std::unordered_map<uint64_t, uint32_t> const& beliefStateToObservationMap,
        storm::storage::Scheduler<BeliefMdpValueType> const& beliefMdpScheduler,
        std::optional<std::vector<storm::storage::Scheduler<typename PomdpModelType::ValueType>>> const& pomdpApproximationSchedulers = std::nullopt,
        PomdpModelType const* preprocessedPomdp = nullptr);

    ObservationBasedFiniteStateController<typename PomdpModelType::ValueType> exportPolicyAsFiniteStateController() const;

    std::shared_ptr<ModelType> exportPolicyAsInducedMarkovChain() const;

   private:
    std::unordered_map<uint64_t, storm::storage::Distribution<typename PomdpModelType::ValueType, uint64_t>> pomdpSchedulerToObservationBasedMap(
        storm::storage::Scheduler<typename PomdpModelType::ValueType> const& scheduler) const;

    PomdpModelType const& pomdp;
    BeliefMdpType const& beliefMdp;
    storm::storage::Scheduler<BeliefMdpValueType> const& beliefMdpScheduler;
    std::unordered_map<uint64_t, uint32_t> const& beliefStateToObservationMap;
    std::optional<std::vector<storm::storage::Scheduler<typename PomdpModelType::ValueType>>> const pomdpApproximationSchedulers;
    PomdpModelType const* preprocessedPomdp;
};
}  // namespace pomdp::policy
}  // namespace storm
