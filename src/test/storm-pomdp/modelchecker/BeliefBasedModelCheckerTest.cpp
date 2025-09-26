#include "storm-config.h"
#include "test/storm_gtest.h"

#include "storm-parsers/api/storm-parsers.h"
#include "storm-pomdp/analysis/FormulaInformation.h"
#include "storm-pomdp/analysis/QualitativeAnalysisOnGraphs.h"
#include "storm-pomdp/beliefs/verification/BeliefBasedModelChecker.h"
#include "storm-pomdp/modelchecker/PreprocessingPomdpValueBoundsModelChecker.h"
#include "storm-pomdp/transformer/GlobalPOMDPSelfLoopEliminator.h"
#include "storm-pomdp/transformer/KnownProbabilityTransformer.h"
#include "storm-pomdp/transformer/MakePOMDPCanonic.h"
#include "storm-pomdp/transformer/MakeStateSetObservationClosed.h"
#include "storm/api/storm.h"
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include "storm/utility/graph.h"

namespace {
enum class PreprocessingType { None, SelfloopReduction, QualitativeReduction, All };

class DefaultDoubleVIEnvironment {
   public:
    typedef double ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration);
        env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-6));
        return env;
    }
    static bool const isExactModelChecking = false;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.12);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::None;
    static uint64_t overApproxResolution() {
        return 2;
    }
};

class SelfloopReductionDefaultDoubleVIEnvironment {
   public:
    typedef double ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration);
        env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-6));
        return env;
    }
    static bool const isExactModelChecking = false;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.12);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::SelfloopReduction;
    static uint64_t overApproxResolution() {
        return 2;
    }
};

class QualitativeReductionDefaultDoubleVIEnvironment {
   public:
    typedef double ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration);
        env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-6));
        return env;
    }
    static bool const isExactModelChecking = false;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.12);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::QualitativeReduction;
    static uint64_t overApproxResolution() {
        return 2;
    }
};

class PreprocessedDefaultDoubleVIEnvironment {
   public:
    typedef double ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration);
        env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-6));
        return env;
    }
    static bool const isExactModelChecking = false;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.12);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::All;
    static uint64_t overApproxResolution() {
        return 2;
    }
};

class FineDoubleVIEnvironment {
   public:
    typedef double ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration);
        env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-6));
        return env;
    }
    static bool const isExactModelChecking = false;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.02);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::None;
    static uint64_t overApproxResolution() {
        return 24;
    }
};

class DefaultDoubleOVIEnvironment {
   public:
    typedef double ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::OptimisticValueIteration);
        env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-6));
        env.solver().setForceSoundness(true);
        return env;
    }
    static bool const isExactModelChecking = false;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.12);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::None;
    static uint64_t overApproxResolution() {
        return 2;
    }
};

class DefaultDoubleSVIEnvironment {
   public:
    typedef double ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::SoundValueIteration);
        env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-6));
        env.solver().setForceSoundness(true);
        return env;
    }
    static bool const isExactModelChecking = false;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.12);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::None;
    static uint64_t overApproxResolution() {
        return 2;
    }
};

class DefaultRationalPIEnvironment {
   public:
    typedef storm::RationalNumber ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::PolicyIteration);
        env.solver().setForceExact(true);
        return env;
    }
    static bool const isExactModelChecking = true;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.12);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::None;
    static uint64_t overApproxResolution() {
        return 2;
    }
};

class PreprocessedDefaultRationalPIEnvironment {
   public:
    typedef storm::RationalNumber ValueType;
    static storm::Environment createEnvironment() {
        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::PolicyIteration);
        env.solver().setForceExact(true);
        return env;
    }
    static bool const isExactModelChecking = true;
    static ValueType precision() {
        return storm::utility::convertNumber<ValueType>(0.12);
    }  // there actually aren't any precision guarantees, but we still want to detect if results are weird.
    static PreprocessingType const preprocessingType = PreprocessingType::All;
    static uint64_t overApproxResolution() {
        return 2;
    }
};

template<typename TestType>
class BeliefBasedModelCheckerTest : public ::testing::Test {
   public:
    typedef typename TestType::ValueType ValueType;
    BeliefBasedModelCheckerTest() : _environment(TestType::createEnvironment()) {}

    void SetUp() override {
#ifndef STORM_HAVE_Z3
        GTEST_SKIP() << "Z3 not available.";
#endif
    }

    storm::Environment const& env() const {
        return _environment;
    }
    ValueType parseNumber(std::string const& str) {
        return storm::utility::convertNumber<ValueType>(str);
    }
    struct Input {
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> model;
        std::shared_ptr<storm::logic::Formula const> formula;
        std::shared_ptr<storm::pomdp::beliefs::PropertyInformation> propertyInfo = std::make_shared<storm::pomdp::beliefs::PropertyInformation>();
    };
    Input buildPrism(std::string const& programFile, std::string const& formulaAsString, std::string const& constantsAsString = "") const {
        // Parse and build input
        storm::prism::Program program = storm::api::parseProgram(programFile);
        program = storm::utility::prism::preprocess(program, constantsAsString);
        Input input;
        input.formula = storm::api::parsePropertiesForPrismProgram(formulaAsString, program).front().getRawFormula();
        input.model = storm::api::buildSparseModel<ValueType>(program, {input.formula})->template as<storm::models::sparse::Pomdp<ValueType>>();

        // Preprocess
        storm::transformer::MakePOMDPCanonic<ValueType> makeCanonic(*input.model);
        input.model = makeCanonic.transform();
        EXPECT_TRUE(input.model->isCanonic());
        if (TestType::preprocessingType == PreprocessingType::SelfloopReduction || TestType::preprocessingType == PreprocessingType::All) {
            storm::transformer::GlobalPOMDPSelfLoopEliminator<ValueType> selfLoopEliminator(*input.model);
            if (selfLoopEliminator.preservesFormula(*input.formula)) {
                input.model = selfLoopEliminator.transform();
            } else {
                EXPECT_TRUE(input.formula->isOperatorFormula());
                EXPECT_TRUE(input.formula->asOperatorFormula().hasOptimalityType());
                bool maximizing = storm::solver::maximize(input.formula->asOperatorFormula().getOptimalityType());
                // Valid reasons for unpreserved formulas:
                EXPECT_TRUE(maximizing || input.formula->isProbabilityOperatorFormula());
                EXPECT_TRUE(!maximizing || input.formula->isRewardOperatorFormula());
            }
        }
        if (TestType::preprocessingType == PreprocessingType::QualitativeReduction || TestType::preprocessingType == PreprocessingType::All) {
            EXPECT_TRUE(input.formula->isOperatorFormula());
            EXPECT_TRUE(input.formula->asOperatorFormula().hasOptimalityType());
            if (input.formula->isProbabilityOperatorFormula() && storm::solver::maximize(input.formula->asOperatorFormula().getOptimalityType())) {
                storm::analysis::QualitativeAnalysisOnGraphs<ValueType> qualitativeAnalysis(*input.model);
                storm::storage::BitVector prob0States = qualitativeAnalysis.analyseProb0(input.formula->asProbabilityOperatorFormula());
                storm::storage::BitVector prob1States = qualitativeAnalysis.analyseProb1(input.formula->asProbabilityOperatorFormula());
                storm::pomdp::transformer::KnownProbabilityTransformer<ValueType> kpt;
                input.model = kpt.transform(*input.model, prob0States, prob1States);
            }
        }
        EXPECT_TRUE(input.model->isCanonic());
        auto formulaInfo = storm::pomdp::analysis::getFormulaInformation(*input.model, *input.formula);
        std::optional<std::string> rewardModelName;
        std::set<uint32_t> targetObservations;
        EXPECT_TRUE(formulaInfo.isNonNestedReachabilityProbability() || formulaInfo.isNonNestedExpectedRewardFormula());
        if (formulaInfo.getTargetStates().observationClosed) {
            targetObservations = formulaInfo.getTargetStates().observations;
        } else {
            storm::transformer::MakeStateSetObservationClosed<ValueType> obsCloser(input.model);
            std::tie(input.model, targetObservations) = obsCloser.transform(formulaInfo.getTargetStates().states);
        }
        if (formulaInfo.isNonNestedReachabilityProbability()) {
            if (!formulaInfo.getSinkStates().empty()) {
                storm::storage::sparse::ModelComponents<ValueType> components;
                components.stateLabeling = input.model->getStateLabeling();
                components.rewardModels = input.model->getRewardModels();
                auto matrix = input.model->getTransitionMatrix();
                matrix.makeRowGroupsAbsorbing(formulaInfo.getSinkStates().states);
                components.transitionMatrix = matrix;
                components.observabilityClasses = input.model->getObservations();
                if (input.model->hasChoiceLabeling()) {
                    components.choiceLabeling = input.model->getChoiceLabeling();
                }
                if (input.model->hasObservationValuations()) {
                    components.observationValuations = input.model->getObservationValuations();
                }
                input.model = std::make_shared<storm::models::sparse::Pomdp<ValueType>>(std::move(components), true);
                auto reachableFromSinkStates =
                    storm::utility::graph::getReachableStates(input.model->getTransitionMatrix(), formulaInfo.getSinkStates().states,
                                                              formulaInfo.getSinkStates().states, ~formulaInfo.getSinkStates().states);
                reachableFromSinkStates &= ~formulaInfo.getSinkStates().states;
                STORM_LOG_THROW(reachableFromSinkStates.empty(), storm::exceptions::NotSupportedException,
                                "There are sink states that can reach non-sink states. This is currently not supported");
            }
        } else {
            // Expected reward formula!
            rewardModelName = formulaInfo.getRewardModelName();
        }

        if (rewardModelName) {
            input.propertyInfo->kind = storm::pomdp::beliefs::PropertyInformation::Kind::ExpectedTotalReachabilityReward;
            input.propertyInfo->rewardModelName = rewardModelName;
        } else {
            input.propertyInfo->kind = storm::pomdp::beliefs::PropertyInformation::Kind::ReachabilityProbability;
        }
        input.propertyInfo->dir = formulaInfo.getOptimizationDirection();
        input.propertyInfo->targetObservations = targetObservations;

        return input;
    }
    ValueType precision() const {
        return TestType::precision();
    }
    uint64_t overApproxResolution() const {
        return TestType::overApproxResolution();
    }
    ValueType modelcheckingPrecision() const {
        if (TestType::isExactModelChecking)
            return storm::utility::zero<ValueType>();
        else
            return storm::utility::convertNumber<ValueType>(1e-6);
    }
    bool isExact() const {
        return TestType::isExactModelChecking;
    }

   private:
    storm::Environment _environment;
};

typedef ::testing::Types<DefaultDoubleVIEnvironment, SelfloopReductionDefaultDoubleVIEnvironment, QualitativeReductionDefaultDoubleVIEnvironment,
                         PreprocessedDefaultDoubleVIEnvironment, FineDoubleVIEnvironment, DefaultDoubleOVIEnvironment, DefaultDoubleSVIEnvironment,
                         DefaultRationalPIEnvironment, PreprocessedDefaultRationalPIEnvironment>
    TestingTypes;

TYPED_TEST_SUITE(BeliefBasedModelCheckerTest, TestingTypes, );

TYPED_TEST(BeliefBasedModelCheckerTest, simple_Pmax) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Pmax=? [F \"goal\" ]", "slippery=0");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;
    ValueType expected = this->parseNumber("7/10");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_GE(overResultValue, expected - this->modelcheckingPrecision());

    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_LE(underResultValue, expected + this->modelcheckingPrecision());
    EXPECT_LE(overResultValue - underResultValue, this->precision())
        << "Result [" << underResultValue << ", " << overResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_Pmin) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Pmin=? [F \"goal\" ]", "slippery=0");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("3/10");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_LE(overResultValue, expected + this->modelcheckingPrecision());

    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_GE(underResultValue, expected - this->modelcheckingPrecision());
    EXPECT_LE(storm::utility::abs<ValueType>(overResultValue - underResultValue), this->precision())
        << "Result [" << overResultValue << ", " << underResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_slippery_Pmax) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Pmax=? [F \"goal\" ]", "slippery=0.4");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;
    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("7/10");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_GE(overResultValue, expected - this->modelcheckingPrecision());
    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();

    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_LE(underResultValue, expected + this->modelcheckingPrecision());
    EXPECT_LE(overResultValue - underResultValue, this->precision())
        << "Result [" << underResultValue << ", " << overResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}
/*
TYPED_TEST(BeliefBasedModelCheckerTest, simple_slippery_Pmin) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Pmin=? [F \"goal\" ]", "slippery=0.4");
storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("3/10");
    if (this->isExact()) {
        // This model's value can only be approximated arbitrarily close but never reached
        // Exact arithmetics will thus not reach the value with absoulute precision either.
        ValueType approxPrecision = storm::utility::convertNumber<ValueType>(1e-5);
        EXPECT_LE(result.lowerBound, expected + approxPrecision);
        EXPECT_GE(result.upperBound, expected - approxPrecision);
    } else {
        EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
        EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    }
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}
*/

TYPED_TEST(BeliefBasedModelCheckerTest, simple_Rmax) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Rmax=? [F s>4 ]", "slippery=0");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("29/50");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_GE(overResultValue, expected - this->modelcheckingPrecision());

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_LE(underResultValue, expected + this->modelcheckingPrecision());
    EXPECT_LE(overResultValue - underResultValue, this->precision())
        << "Result [" << underResultValue << ", " << overResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_Rmin) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Rmin=? [F s>4 ]", "slippery=0");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("19/50");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_LE(overResultValue, expected + this->modelcheckingPrecision());

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_GE(underResultValue, expected - this->modelcheckingPrecision());
    EXPECT_LE(storm::utility::abs<ValueType>(overResultValue - underResultValue), this->precision())
        << "Result [" << overResultValue << ", " << underResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_slippery_Rmax) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Rmax=? [F s>4 ]", "slippery=0.4");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("29/30");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_GE(overResultValue, expected - this->modelcheckingPrecision());

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_LE(underResultValue, expected + this->modelcheckingPrecision());
    EXPECT_LE(overResultValue - underResultValue, this->precision())
        << "Result [" << underResultValue << ", " << overResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_slippery_Rmin) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Rmin=? [F s>4 ]", "slippery=0.4");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("19/30");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_LE(overResultValue, expected + this->modelcheckingPrecision());

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_GE(underResultValue, expected - this->modelcheckingPrecision());
    EXPECT_LE(storm::utility::abs<ValueType>(overResultValue - underResultValue), this->precision())
        << "Result [" << overResultValue << ", " << underResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, maze2_Rmin) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/maze2.prism", "Rmin=? [F \"goal\"]", "sl=0");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("74/91");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    // EXPECT_LE(overResultValue, expected + this->modelcheckingPrecision());

    options.maxExplorationSize = 22;
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    // EXPECT_GE(underResultValue, expected - this->modelcheckingPrecision());
    // EXPECT_LE(storm::utility::abs<ValueType>(overResultValue - underResultValue), this->precision())
    //     << "Result [" << overResultValue << ", " << underResultValue
    //     << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, maze2_Rmax) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/maze2.prism", "Rmax=? [F \"goal\"]", "sl=0");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_TRUE(storm::utility::isInfinity(overResultValue));

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_TRUE(storm::utility::isInfinity(underResultValue));
}

TYPED_TEST(BeliefBasedModelCheckerTest, maze2_slippery_Rmin) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/maze2.prism", "Rmin=? [F \"goal\"]", "sl=0.075");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("80/91");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_LE(overResultValue, expected + this->modelcheckingPrecision());

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_GE(underResultValue, expected - this->modelcheckingPrecision());
    EXPECT_LE(storm::utility::abs<ValueType>(overResultValue - underResultValue), this->precision())
        << "Result [" << overResultValue << ", " << underResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, maze2_slippery_Rmax) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/maze2.prism", "Rmax=? [F \"goal\"]", "sl=0.075");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_TRUE(storm::utility::isInfinity(overResultValue));

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_TRUE(storm::utility::isInfinity(underResultValue));
}

TYPED_TEST(BeliefBasedModelCheckerTest, refuel_Pmax) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/refuel.prism", "Pmax=?[\"notbad\" U \"goal\"]", "N=4");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("38/155");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_GE(overResultValue, expected - this->modelcheckingPrecision());

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_LE(underResultValue, expected + this->modelcheckingPrecision());
    EXPECT_LE(overResultValue - underResultValue, this->precision())
        << "Result [" << underResultValue << ", " << overResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, refuel_Pmin) {
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/refuel.prism", "Pmin=?[\"notbad\" U \"goal\"]", "N=4");
    storm::pomdp::beliefs::BeliefBasedModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(*data.model);
    storm::pomdp::modelchecker::PreprocessingPomdpValueBoundsModelChecker<ValueType> preprocessChecker(*data.model);

    storm::pomdp::storage::PreprocessingPomdpValueBounds<ValueType> precomputedPomdpValueBounds = preprocessChecker.getValueBounds(this->env(), *data.formula);
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<ValueType> options;
    options.buildChoiceLabeling = false;
    options.generatePolicy = false;
    options.explorationQueueOrder = storm::pomdp::beliefs::ExplorationQueueOrder::FIFO;

    ValueType overResultValue;
    ValueType underResultValue;
    bool completedOverExploration;
    bool completedUnderExploration;

    ValueType expected = this->parseNumber("0");
    std::tie(overResultValue, completedOverExploration) =
        checker.checkDiscretize(this->env(), *data.propertyInfo, options, this->overApproxResolution(), true, precomputedPomdpValueBounds);
    EXPECT_LE(overResultValue, expected + this->modelcheckingPrecision());

    options.maxExplorationSize = data.model->getNumberOfStates() * data.model->getMaxNrStatesWithSameObservation();
    std::tie(underResultValue, completedUnderExploration) = checker.checkUnfold(this->env(), *data.propertyInfo, options, precomputedPomdpValueBounds);
    EXPECT_GE(underResultValue, expected - this->modelcheckingPrecision());
    EXPECT_LE(storm::utility::abs<ValueType>(overResultValue - underResultValue), this->precision())
        << "Result [" << overResultValue << ", " << underResultValue
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}
/*
#if defined STORM_HAVE_Z3_OPTIMIZE

TYPED_TEST(BeliefBasedModelCheckerTest, simple_Pmax_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Pmax=? [F \"goal\" ]", "slippery=0");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("7/10");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_Pmin_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Pmin=? [F \"goal\" ]", "slippery=0");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("3/10");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_slippery_Pmax_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Pmax=? [F \"goal\" ]", "slippery=0.4");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("7/10");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_slippery_Pmin_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Pmin=? [F \"goal\" ]", "slippery=0.4");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("3/10");
    if (this->isExact()) {
        // This model's value can only be approximated arbitrarily close but never reached
        // Exact arithmetics will thus not reach the value with absoulute precision either.
        ValueType approxPrecision = storm::utility::convertNumber<ValueType>(1e-4);
        EXPECT_LE(result.lowerBound, expected + approxPrecision);
        EXPECT_GE(result.upperBound, expected - approxPrecision);
    } else {
        EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision() * 10);
        EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision() * 10);
    }
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_Rmax_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Rmax=? [F s>4 ]", "slippery=0");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("29/50");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_Rmin_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Rmin=? [F s>4 ]", "slippery=0");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("19/50");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_slippery_Rmax_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Rmax=? [F s>4 ]", "slippery=0.4");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("29/30");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, simple_slippery_Rmin_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/simple.prism", "Rmin=? [F s>4 ]", "slippery=0.4");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("19/30");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, maze2_Rmin_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/maze2.prism", "Rmin=? [F \"goal\"]", "sl=0");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("74/91");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    // Use relative difference of bounds for this one
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, maze2_Rmax_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/maze2.prism", "Rmax=? [F \"goal\"]", "sl=0");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    EXPECT_TRUE(storm::utility::isInfinity(result.lowerBound));
    EXPECT_TRUE(storm::utility::isInfinity(result.upperBound));
}

TYPED_TEST(BeliefBasedModelCheckerTest, maze2_slippery_Rmin_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/maze2.prism", "Rmin=? [F \"goal\"]", "sl=0.075");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("80/91");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    // Use relative difference of bounds for this one
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, maze2_slippery_Rmax_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/maze2.prism", "Rmax=? [F \"goal\"]", "sl=0.075");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    EXPECT_TRUE(storm::utility::isInfinity(result.lowerBound));
    EXPECT_TRUE(storm::utility::isInfinity(result.upperBound));
}

TYPED_TEST(BeliefBasedModelCheckerTest, refuel_Pmax_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/refuel.prism", "Pmax=?[\"notbad\" U \"goal\"]", "N=4");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("38/155");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    // Use relative difference of bounds for this one
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}

TYPED_TEST(BeliefBasedModelCheckerTest, refuel_Pmin_Clip) {
    if (!storm::test::z3AtLeastVersion(4, 8, 5)) {
        GTEST_SKIP() << "Test disabled since it triggers a bug in the installed version of z3.";
    }
    typedef typename TestFixture::ValueType ValueType;

    auto data = this->buildPrism(STORM_TEST_RESOURCES_DIR "/pomdp/refuel.prism", "Pmin=?[\"notbad\" U \"goal\"]", "N=4");
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType>> checker(data.model, this->optionsWithClipping());
    auto result = checker.check(this->env(), *data.formula);

    ValueType expected = this->parseNumber("0");
    EXPECT_LE(result.lowerBound, expected + this->modelcheckingPrecision());
    EXPECT_GE(result.upperBound, expected - this->modelcheckingPrecision());
    // Use relative difference of bounds for this one
    EXPECT_LE(result.diff(), this->precision())
        << "Result [" << result.lowerBound << ", " << result.upperBound
        << "] is not precise enough. If (only) this fails, the result bounds are still correct, but they might be unexpectedly imprecise.\n";
}
#endif  // defined STORM_HAVE_Z3_OPTIMIZE*/

}  // namespace
