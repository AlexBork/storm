#include "test/storm_gtest.h"

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/models/sparse/Ctmc.h"
#include "storm/models/sparse/Dtmc.h"
#include "storm/models/sparse/MarkovAutomaton.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/transformer/SparseRationalModelToDoubleTransformer.h"

namespace {

constexpr double normalizationPrecision = 1e-17;
constexpr double standardPrecision = 1e-9;

storm::storage::SparseMatrix<storm::RationalNumber> buildProbabilityMatrix() {
    storm::storage::SparseMatrixBuilder<storm::RationalNumber> builder(3, 3, 5);
    builder.addNextValue(0, 0, storm::RationalNumber("1/5"));
    builder.addNextValue(0, 1, storm::RationalNumber("3/5"));
    builder.addNextValue(0, 2, storm::RationalNumber("1/5"));
    builder.addNextValue(1, 1, storm::RationalNumber(1));
    builder.addNextValue(2, 2, storm::RationalNumber(1));
    return builder.build();
}

TEST(SparseRationalModelToDoubleTransformerTest, NormalizesDtmcOutsideConfiguredPrecision) {
    auto const matrix = buildProbabilityMatrix();

    auto const inputModel = std::make_shared<storm::models::sparse::Dtmc<storm::RationalNumber>>(std::move(matrix), storm::models::sparse::StateLabeling(3));
    auto const result = storm::transformer::sparseRationalModelToDouble(inputModel, normalizationPrecision)->as<storm::models::sparse::Dtmc<double>>();

    EXPECT_TRUE(result->getTransitionMatrix().isProbabilistic(normalizationPrecision));
    EXPECT_DOUBLE_EQ(1.0, result->getTransitionMatrix().getRowSum(0));
}

TEST(SparseRationalModelToDoubleTransformerTest, PreservesDtmcWithinConfiguredPrecision) {
    auto matrix = buildProbabilityMatrix();
    auto convertedMatrix = matrix.toValueType<double>();
    ASSERT_TRUE(convertedMatrix.isProbabilistic(standardPrecision));
    ASSERT_NE(1.0, convertedMatrix.getRowSum(0));

    auto inputModel = std::make_shared<storm::models::sparse::Dtmc<storm::RationalNumber>>(std::move(matrix), storm::models::sparse::StateLabeling(3));
    auto result = storm::transformer::sparseRationalModelToDouble(inputModel, standardPrecision)->as<storm::models::sparse::Dtmc<double>>();

    EXPECT_EQ(convertedMatrix, result->getTransitionMatrix());
}

TEST(SparseRationalModelToDoubleTransformerTest, PreservesCtmcRates) {
    storm::storage::SparseMatrixBuilder<storm::RationalNumber> builder(2, 2, 4);
    builder.addNextValue(0, 0, storm::RationalNumber(2));
    builder.addNextValue(0, 1, storm::RationalNumber(1));
    builder.addNextValue(1, 0, storm::RationalNumber(1));
    builder.addNextValue(1, 1, storm::RationalNumber(3));
    auto matrix = builder.build();
    auto convertedMatrix = matrix.toValueType<double>();

    auto inputModel = std::make_shared<storm::models::sparse::Ctmc<storm::RationalNumber>>(std::move(matrix), storm::models::sparse::StateLabeling(2));
    auto result = storm::transformer::sparseRationalModelToDouble(inputModel, standardPrecision)->as<storm::models::sparse::Ctmc<double>>();

    EXPECT_EQ(convertedMatrix, result->getTransitionMatrix());
    EXPECT_EQ((std::vector<double>{3.0, 4.0}), result->getExitRateVector());
}

TEST(SparseRationalModelToDoubleTransformerTest, NormalizesMarkovAutomatonProbabilities) {
    storm::storage::SparseMatrixBuilder<storm::RationalNumber> builder(3, 3, 5, true, true, 3);
    builder.newRowGroup(0);
    builder.addNextValue(0, 0, storm::RationalNumber(1));
    builder.addNextValue(0, 1, storm::RationalNumber(3));
    builder.addNextValue(0, 2, storm::RationalNumber(1));
    builder.newRowGroup(1);
    builder.addNextValue(1, 1, storm::RationalNumber(1));
    builder.newRowGroup(2);
    builder.addNextValue(2, 2, storm::RationalNumber(1));
    storm::storage::BitVector markovianStates(3, false);
    markovianStates.set(0);

    auto inputModel = std::make_shared<storm::models::sparse::MarkovAutomaton<storm::RationalNumber>>(builder.build(), storm::models::sparse::StateLabeling(3),
                                                                                                      std::move(markovianStates));
    ASSERT_FALSE(inputModel->getTransitionMatrix().toValueType<double>().isProbabilistic(normalizationPrecision));
    auto result = storm::transformer::sparseRationalModelToDouble(inputModel, normalizationPrecision)->as<storm::models::sparse::MarkovAutomaton<double>>();

    EXPECT_TRUE(result->getTransitionMatrix().isProbabilistic(normalizationPrecision));
    EXPECT_DOUBLE_EQ(1.0, result->getTransitionMatrix().getRowSum(0));
    EXPECT_EQ(inputModel->getMarkovianStates(), result->getMarkovianStates());
    EXPECT_EQ((std::vector<double>{5.0, 0.0, 0.0}), result->getExitRates());
}

}  // namespace
