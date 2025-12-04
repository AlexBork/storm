#pragma once

namespace storm::pomdp::beliefs {
template<typename BeliefMdpValueType>
struct BeliefBasedModelCheckerResult {
    BeliefMdpValueType value;
    bool completedExploration;
};
}  // namespace storm::pomdp::beliefs