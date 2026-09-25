#include <cstdint>
#include <utility>
#include <vector>

#include "test/storm_gtest.h"

#include "storm-pomdp/transformer/ToStateBasedObservationTransformer.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StateLabeling.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/sparse/ModelComponents.h"

namespace {

std::shared_ptr<storm::models::sparse::Mdp<double>> buildMdp() {
    storm::storage::SparseMatrixBuilder<double> matrixBuilder(0, 0, 0, false, true);
    matrixBuilder.newRowGroup(0);
    matrixBuilder.addNextValue(0, 1, 1.0);
    matrixBuilder.addNextValue(1, 1, 1.0);
    matrixBuilder.newRowGroup(2);
    matrixBuilder.addNextValue(2, 2, 1.0);
    matrixBuilder.addNextValue(3, 2, 1.0);
    matrixBuilder.newRowGroup(4);
    matrixBuilder.addNextValue(4, 2, 1.0);

    storm::models::sparse::StateLabeling labeling(3);
    labeling.addLabel("init");
    labeling.addLabelToState("init", 0);
    storm::storage::sparse::ModelComponents<double> components(matrixBuilder.build(), std::move(labeling));
    return std::make_shared<storm::models::sparse::Mdp<double>>(std::move(components));
}

}  // namespace

TEST(ToStateBasedObservationTransformer, PassesLocalActionIndicesToCallback) {
    auto const mdp = buildMdp();
    std::vector<std::pair<uint64_t, uint64_t>> observedStateActions;

    auto const result = storm::pomdp::transformer::ToStateBasedObservationTransformer<double>::transform(
        *mdp,
        [&observedStateActions](uint64_t state, uint64_t action, uint64_t) {
            observedStateActions.emplace_back(state, action);
            return 0;
        },
        0);

    EXPECT_NE(nullptr, result);
    EXPECT_EQ((std::vector<std::pair<uint64_t, uint64_t>>{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}}), observedStateActions);
}
