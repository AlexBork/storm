#pragma once

#include "BeliefBasedModelCheckerOptions.h"
#include "storm-pomdp/beliefs/verification/PropertyInformation.h"
#include "storm-pomdp/modelchecker/BeliefExplorationPomdpModelCheckerOptions.h"
#include "storm-pomdp/storage/BeliefExplorationBounds.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/models/sparse/Pomdp.h"

namespace storm {
class Environment;

namespace pomdp::beliefs {

template<typename PomdpModelType, typename BeliefValueType = typename PomdpModelType::ValueType,
         typename BeliefMdpValueType = typename PomdpModelType::ValueType>
class BeliefBasedModelChecker {
   public:
    using PomdpValueType = PomdpModelType::ValueType;
    explicit BeliefBasedModelChecker(PomdpModelType const& pomdp);

    BeliefBasedModelChecker(PomdpModelType const& pomdp, storm::models::sparse::Pomdp<storm::RationalNumber> const& exactInputPomdp);

    std::pair<BeliefMdpValueType, bool> checkUnfold(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                    BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                    storage::BeliefExplorationBounds<PomdpValueType> const& valueBounds);

    std::pair<BeliefMdpValueType, bool> checkDiscretize(storm::Environment const& env, PropertyInformation const& propertyInformation,
                                                        storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                        uint64_t resolution, bool useDynamic,
                                                        storage::BeliefExplorationBounds<PomdpValueType> const& valueBounds);

   private:
    PomdpModelType const& inputPomdp;
    // Optional version of the input POMDP used for computing exact beliefs if the input POMDP uses floating point numbers.
    std::optional<storm::models::sparse::Pomdp<storm::RationalNumber>> exactPomdp = std::nullopt;
};
}  // namespace pomdp::beliefs
}  // namespace storm