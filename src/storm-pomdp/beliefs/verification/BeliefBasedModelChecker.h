#pragma once

#include "BeliefBasedModelCheckerOptions.h"
#include "storm-pomdp/beliefs/verification/PropertyInformation.h"
#include "storm-pomdp/modelchecker/BeliefExplorationPomdpModelCheckerOptions.h"
#include "storm-pomdp/storage/BeliefExplorationBounds.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/models/sparse/Pomdp.h"

#include <optional>

namespace storm {
class Environment;

namespace pomdp::beliefs {

template<typename PomdpModelType, typename BeliefValueType = typename PomdpModelType::ValueType,
         typename BeliefMdpValueType = typename PomdpModelType::ValueType>
class BeliefBasedModelChecker {
   public:
    using PomdpValueType = PomdpModelType::ValueType;

    struct RunStatistics {
        bool available = false;
        bool completedExploration = false;
        uint64_t discoveredBeliefs = 0;
        uint64_t exploredBeliefs = 0;
        uint64_t beliefMdpStates = 0;
        uint64_t beliefMdpChoices = 0;
        uint64_t beliefMdpTransitions = 0;
        std::optional<uint64_t> processedMdpStates;
        std::optional<uint64_t> processedMdpChoices;
        std::optional<uint64_t> processedMdpTransitions;
        uint64_t explorationTimeMilliseconds = 0;
        uint64_t beliefMdpBuildTimeMilliseconds = 0;
        uint64_t beliefMdpAnalysisTimeMilliseconds = 0;
    };

    explicit BeliefBasedModelChecker(PomdpModelType const& pomdp);

    BeliefBasedModelChecker(PomdpModelType const& pomdp, storm::models::sparse::Pomdp<storm::RationalNumber> const& exactInputPomdp);

    std::pair<BeliefMdpValueType, bool> checkUnfold(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                    BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                    storage::BeliefExplorationBounds<PomdpValueType> const& valueBounds);

    std::pair<BeliefMdpValueType, bool> checkDiscretize(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                        storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                        uint64_t resolution, bool useDynamic,
                                                        storage::BeliefExplorationBounds<PomdpValueType> const& valueBounds);

    std::pair<BeliefMdpValueType, bool> checkRewardAwareUnfold(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                               storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                               storage::BeliefExplorationBounds<typename PomdpModelType::ValueType> const& valueBounds,
                                                               std::vector<std::string> const& relevantRewardModelNames = {});

    std::pair<BeliefMdpValueType, bool> checkRewardAwareDiscretize(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                                   storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                                   uint64_t resolution, bool useDynamic,
                                                                   storage::BeliefExplorationBounds<typename PomdpModelType::ValueType> const& valueBounds,
                                                                   std::vector<std::string> const& relevantRewardModelNames = {});

    RunStatistics const& getLastRunStatistics() const;

   private:
    PomdpModelType const& inputPomdp;
    // Optional version of the input POMDP used for computing exact beliefs if the input POMDP uses floating point numbers.
    std::optional<storm::models::sparse::Pomdp<storm::RationalNumber>> exactPomdp = std::nullopt;
    RunStatistics lastRunStatistics;
};
}  // namespace pomdp::beliefs
}  // namespace storm
