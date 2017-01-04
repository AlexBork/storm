 #include "storm/modelchecker/prctl/helper/SparseMdpPrctlHelper.h"

#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"

#include "storm/models/sparse/StandardRewardModel.h"

#include "storm/storage/MaximalEndComponentDecomposition.h"

#include "storm/utility/macros.h"
#include "storm/utility/vector.h"
#include "storm/utility/graph.h"

#include "storm/storage/expressions/Variable.h"
#include "storm/storage/expressions/Expression.h"
#include "storm/storage/TotalScheduler.h"

#include "storm/solver/MinMaxLinearEquationSolver.h"
#include "storm/solver/LpSolver.h"

#include "storm/exceptions/InvalidStateException.h"
#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/exceptions/InvalidSettingsException.h"
#include "storm/exceptions/IllegalFunctionCallException.h"

namespace storm {
    namespace modelchecker {
        namespace helper {

            template<typename ValueType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeBoundedUntilProbabilities(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, uint_fast64_t stepBound, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                std::vector<ValueType> result(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                
                // Determine the states that have 0 probability of reaching the target states.
                storm::storage::BitVector maybeStates;
                if (dir == OptimizationDirection::Minimize) {
                    maybeStates = storm::utility::graph::performProbGreater0A(transitionMatrix, transitionMatrix.getRowGroupIndices(), backwardTransitions, phiStates, psiStates, true, stepBound);
                } else {
                    maybeStates = storm::utility::graph::performProbGreater0E(backwardTransitions, phiStates, psiStates, true, stepBound);
                }
                maybeStates &= ~psiStates;
                STORM_LOG_INFO("Found " << maybeStates.getNumberOfSetBits() << " 'maybe' states.");
                
                if (!maybeStates.empty()) {
                    // We can eliminate the rows and columns from the original transition probability matrix that have probability 0.
                    storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, maybeStates, maybeStates, false);
                    std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(maybeStates, psiStates);
                    
                    // Create the vector with which to multiply.
                    std::vector<ValueType> subresult(maybeStates.getNumberOfSetBits());
                    
                    std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(std::move(submatrix));
                    solver->repeatedMultiply(dir, subresult, &b, stepBound);
                    
                    // Set the values of the resulting vector accordingly.
                    storm::utility::vector::setVectorValues(result, maybeStates, subresult);
                }
                storm::utility::vector::setVectorValues(result, psiStates, storm::utility::one<ValueType>());
                
                return result;
            }

            template<typename ValueType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeNextProbabilities(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const& nextStates, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {

                // Create the vector with which to multiply and initialize it correctly.
                std::vector<ValueType> result(transitionMatrix.getRowGroupCount());
                storm::utility::vector::setVectorValues(result, nextStates, storm::utility::one<ValueType>());
                
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(transitionMatrix);
                solver->repeatedMultiply(dir, result, nullptr, 1);
                
                return result;
            }
            
            template<typename ValueType>
            MDPSparseModelCheckingHelperReturnType<ValueType> SparseMdpPrctlHelper<ValueType>::computeUntilProbabilities(storm::solver::SolveGoal const& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, bool produceScheduler, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                STORM_LOG_THROW(!(qualitative && produceScheduler), storm::exceptions::InvalidSettingsException, "Cannot produce scheduler when performing qualitative model checking only.");
                     
                // We need to identify the states which have to be taken out of the matrix, i.e.
                // all states that have probability 0 and 1 of satisfying the until-formula.
                std::pair<storm::storage::BitVector, storm::storage::BitVector> statesWithProbability01;
                if (goal.minimize()) {
                    statesWithProbability01 = storm::utility::graph::performProb01Min(transitionMatrix, transitionMatrix.getRowGroupIndices(), backwardTransitions, phiStates, psiStates);
                } else {
                    statesWithProbability01 = storm::utility::graph::performProb01Max(transitionMatrix, transitionMatrix.getRowGroupIndices(), backwardTransitions, phiStates, psiStates);
                }
                storm::storage::BitVector statesWithProbability0 = std::move(statesWithProbability01.first);
                storm::storage::BitVector statesWithProbability1 = std::move(statesWithProbability01.second);
                storm::storage::BitVector maybeStates = ~(statesWithProbability0 | statesWithProbability1);
                STORM_LOG_INFO("Found " << statesWithProbability0.getNumberOfSetBits() << " 'no' states.");
                STORM_LOG_INFO("Found " << statesWithProbability1.getNumberOfSetBits() << " 'yes' states.");
                STORM_LOG_INFO("Found " << maybeStates.getNumberOfSetBits() << " 'maybe' states.");
                
                // Create resulting vector.
                std::vector<ValueType> result(transitionMatrix.getRowGroupCount());
                
                // Set values of resulting vector that are known exactly.
                storm::utility::vector::setVectorValues<ValueType>(result, statesWithProbability0, storm::utility::zero<ValueType>());
                storm::utility::vector::setVectorValues<ValueType>(result, statesWithProbability1, storm::utility::one<ValueType>());

                // If requested, we will produce a scheduler.
                std::unique_ptr<storm::storage::TotalScheduler> scheduler;
                if (produceScheduler) {
                    scheduler = std::make_unique<storm::storage::TotalScheduler>(transitionMatrix.getRowGroupCount());
                }
                
                // Check whether we need to compute exact probabilities for some states.
                if (qualitative) {
                    // Set the values for all maybe-states to 0.5 to indicate that their probability values are neither 0 nor 1.
                    storm::utility::vector::setVectorValues<ValueType>(result, maybeStates, storm::utility::convertNumber<ValueType>(0.5));
                } else {
                    if (!maybeStates.empty()) {
                        // In this case we have have to compute the probabilities.
                        
                        // First, we can eliminate the rows and columns from the original transition probability matrix for states
                        // whose probabilities are already known.
                        storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, maybeStates, maybeStates, false);
                        
                        // Prepare the right-hand side of the equation system. For entry i this corresponds to
                        // the accumulated probability of going from state i to some 'yes' state.
                        std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(maybeStates, statesWithProbability1);
                        
                        MDPSparseModelCheckingHelperReturnType<ValueType> resultForMaybeStates = computeUntilProbabilitiesOnlyMaybeStates(goal, submatrix, b, produceScheduler, minMaxLinearEquationSolverFactory);
                        
                        // Set values of resulting vector according to result.
                        storm::utility::vector::setVectorValues<ValueType>(result, maybeStates, resultForMaybeStates.values);

                        if (produceScheduler) {
                            storm::storage::Scheduler const& subscheduler = *resultForMaybeStates.scheduler;
                            uint_fast64_t currentSubState = 0;
                            for (auto maybeState : maybeStates) {
                                scheduler->setChoice(maybeState, subscheduler.getChoice(currentSubState));
                                ++currentSubState;
                            }
                        }
                    }
                }
                
                // Finally, if we need to produce a scheduler, we also need to figure out the parts of the scheduler for
                // the states with probability 0 or 1 (depending on whether we maximize or minimize).
                if (produceScheduler) {
                    storm::storage::PartialScheduler relevantQualitativeScheduler;
                    if (goal.minimize()) {
                        relevantQualitativeScheduler = storm::utility::graph::computeSchedulerProb0E(statesWithProbability0, transitionMatrix);
                    } else {
                        relevantQualitativeScheduler = storm::utility::graph::computeSchedulerProb1E(statesWithProbability1, transitionMatrix, backwardTransitions, phiStates, psiStates);
                    }
                    for (auto const& stateChoicePair : relevantQualitativeScheduler) {
                        scheduler->setChoice(stateChoicePair.first, stateChoicePair.second);
                    }
                }
                
                return MDPSparseModelCheckingHelperReturnType<ValueType>(std::move(result), std::move(scheduler));
            }
            
            template<typename ValueType>
            MDPSparseModelCheckingHelperReturnType<ValueType> SparseMdpPrctlHelper<ValueType>::computeUntilProbabilitiesOnlyMaybeStates(storm::solver::SolveGoal const& goal, storm::storage::SparseMatrix<ValueType> const& submatrix, std::vector<ValueType> const& b, bool produceScheduler, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                // If requested, we will produce a scheduler.
                std::unique_ptr<storm::storage::TotalScheduler> scheduler;
                
                // Create vector for results for maybe states.
                std::vector<ValueType> x(submatrix.getRowGroupCount());
                
                // Solve the corresponding system of equations.
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = storm::solver::configureMinMaxLinearEquationSolver(goal, minMaxLinearEquationSolverFactory, submatrix);
                solver->setTrackScheduler(produceScheduler);
                solver->solveEquations(x, b);
                
                if (produceScheduler) {
                    scheduler = solver->getScheduler();
                }
                
                return MDPSparseModelCheckingHelperReturnType<ValueType>(std::move(x), std::move(scheduler));
            }

            template<typename ValueType>
            MDPSparseModelCheckingHelperReturnType<ValueType> SparseMdpPrctlHelper<ValueType>::computeUntilProbabilities(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, bool produceScheduler, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                storm::solver::SolveGoal goal(dir);
                return std::move(computeUntilProbabilities(goal, transitionMatrix, backwardTransitions, phiStates, psiStates, qualitative, produceScheduler, minMaxLinearEquationSolverFactory));
            }
           
            template<typename ValueType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeGloballyProbabilities(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& psiStates, bool qualitative, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory, bool useMecBasedTechnique) {
                if (useMecBasedTechnique) {
                    storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(transitionMatrix, backwardTransitions, psiStates);
                    storm::storage::BitVector statesInPsiMecs(transitionMatrix.getRowGroupCount());
                    for (auto const& mec : mecDecomposition) {
                        for (auto const& stateActionsPair : mec) {
                            statesInPsiMecs.set(stateActionsPair.first, true);
                        }
                    }
                    
                    return std::move(computeUntilProbabilities(dir, transitionMatrix, backwardTransitions, psiStates, statesInPsiMecs, qualitative, false, minMaxLinearEquationSolverFactory).values);
                } else {
                    std::vector<ValueType> result = computeUntilProbabilities(dir == OptimizationDirection::Minimize ? OptimizationDirection::Maximize : OptimizationDirection::Minimize, transitionMatrix, backwardTransitions, storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true), ~psiStates, qualitative, false, minMaxLinearEquationSolverFactory).values;
                    for (auto& element : result) {
                        element = storm::utility::one<ValueType>() - element;
                    }
                    return std::move(result);
                }
            }
            
            template<typename ValueType>
            template<typename RewardModelType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeInstantaneousRewards(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, RewardModelType const& rewardModel, uint_fast64_t stepCount, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {

                // Only compute the result if the model has a state-based reward this->getModel().
                STORM_LOG_THROW(rewardModel.hasStateRewards(), storm::exceptions::InvalidPropertyException, "Missing reward model for formula. Skipping formula.");
                
                // Initialize result to state rewards of the this->getModel().
                std::vector<ValueType> result(rewardModel.getStateRewardVector());
                
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(transitionMatrix);
                solver->repeatedMultiply(dir, result, nullptr, stepCount);
                
                return result;
            }
            
            template<typename ValueType>
            template<typename RewardModelType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeCumulativeRewards(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, RewardModelType const& rewardModel, uint_fast64_t stepBound, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {

                // Only compute the result if the model has at least one reward this->getModel().
                STORM_LOG_THROW(!rewardModel.empty(), storm::exceptions::InvalidPropertyException, "Missing reward model for formula. Skipping formula.");
                
                // Compute the reward vector to add in each step based on the available reward models.
                std::vector<ValueType> totalRewardVector = rewardModel.getTotalRewardVector(transitionMatrix);
                
                // Initialize result to either the state rewards of the model or the null vector.
                std::vector<ValueType> result;
                if (rewardModel.hasStateRewards()) {
                    result = rewardModel.getStateRewardVector();
                } else {
                    result.resize(transitionMatrix.getRowGroupCount());
                }
                
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(transitionMatrix);
                solver->repeatedMultiply(dir, result, &totalRewardVector, stepBound);
                
                return result;
            }
            

            template<typename ValueType>
            template<typename RewardModelType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeReachabilityRewards(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, RewardModelType const& rewardModel, storm::storage::BitVector const& targetStates, bool qualitative, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                // Only compute the result if the model has at least one reward this->getModel().
                STORM_LOG_THROW(!rewardModel.empty(), storm::exceptions::InvalidPropertyException, "Missing reward model for formula. Skipping formula.");
                return computeReachabilityRewardsHelper(dir, transitionMatrix, backwardTransitions,
                                                        [&rewardModel] (uint_fast64_t rowCount, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const& maybeStates) {
                                                            return rewardModel.getTotalRewardVector(rowCount, transitionMatrix, maybeStates);
                                                        },
                                                        targetStates, qualitative, minMaxLinearEquationSolverFactory);
            }
            
#ifdef STORM_HAVE_CARL
            template<typename ValueType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeReachabilityRewards(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::models::sparse::StandardRewardModel<storm::Interval> const& intervalRewardModel, bool lowerBoundOfIntervals, storm::storage::BitVector const& targetStates, bool qualitative, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                // Only compute the result if the reward model is not empty.
                STORM_LOG_THROW(!intervalRewardModel.empty(), storm::exceptions::InvalidPropertyException, "Missing reward model for formula. Skipping formula.");
                return computeReachabilityRewardsHelper(dir, transitionMatrix, backwardTransitions, \
                                                        [&] (uint_fast64_t rowCount, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const& maybeStates) {
                                                            std::vector<ValueType> result;
                                                            result.reserve(rowCount);
                                                            std::vector<storm::Interval> subIntervalVector = intervalRewardModel.getTotalRewardVector(rowCount, transitionMatrix, maybeStates);
                                                            for (auto const& interval : subIntervalVector) {
                                                                result.push_back(lowerBoundOfIntervals ? interval.lower() : interval.upper());
                                                            }
                                                            return result;
                                                        }, \
                                                        targetStates, qualitative, minMaxLinearEquationSolverFactory);
            }
            
            template<>
            std::vector<storm::RationalNumber> SparseMdpPrctlHelper<storm::RationalNumber>::computeReachabilityRewards(OptimizationDirection, storm::storage::SparseMatrix<storm::RationalNumber> const&, storm::storage::SparseMatrix<storm::RationalNumber> const&, storm::models::sparse::StandardRewardModel<storm::Interval> const&, bool, storm::storage::BitVector const&, bool, storm::solver::MinMaxLinearEquationSolverFactory<storm::RationalNumber> const&) {
                STORM_LOG_THROW(false, storm::exceptions::IllegalFunctionCallException, "Computing reachability rewards is unsupported for this data type.");
            }
#endif
            
            template<typename ValueType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeReachabilityRewardsHelper(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::function<std::vector<ValueType>(uint_fast64_t, storm::storage::SparseMatrix<ValueType> const&, storm::storage::BitVector const&)> const& totalStateRewardVectorGetter, storm::storage::BitVector const& targetStates, bool qualitative, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                
                std::vector<uint_fast64_t> const& nondeterministicChoiceIndices = transitionMatrix.getRowGroupIndices();
                
                // Determine which states have a reward of infinity by definition.
                storm::storage::BitVector infinityStates;
                storm::storage::BitVector trueStates(transitionMatrix.getRowGroupCount(), true);
                if (dir == OptimizationDirection::Minimize) {
                    STORM_LOG_WARN("Results of reward computation may be too low, because of zero-reward loops.");
                    infinityStates = storm::utility::graph::performProb1E(transitionMatrix, nondeterministicChoiceIndices, backwardTransitions, trueStates, targetStates);
                } else {
                    infinityStates = storm::utility::graph::performProb1A(transitionMatrix, nondeterministicChoiceIndices, backwardTransitions, trueStates, targetStates);
                }
                infinityStates.complement();
                storm::storage::BitVector maybeStates = ~targetStates & ~infinityStates;
                STORM_LOG_INFO("Found " << infinityStates.getNumberOfSetBits() << " 'infinity' states.");
                STORM_LOG_INFO("Found " << targetStates.getNumberOfSetBits() << " 'target' states.");
                STORM_LOG_INFO("Found " << maybeStates.getNumberOfSetBits() << " 'maybe' states.");
                
                // Create resulting vector.
                std::vector<ValueType> result(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                
                // Check whether we need to compute exact rewards for some states.
                if (qualitative) {
                    STORM_LOG_INFO("The rewards for the initial states were determined in a preprocessing step. No exact rewards were computed.");
                    // Set the values for all maybe-states to 1 to indicate that their reward values
                    // are neither 0 nor infinity.
                    storm::utility::vector::setVectorValues<ValueType>(result, maybeStates, storm::utility::one<ValueType>());
                } else {
                    if (!maybeStates.empty()) {
                        // In this case we have to compute the reward values for the remaining states.
                        
                        // We can eliminate the rows and columns from the original transition probability matrix for states
                        // whose reward values are already known.
                        storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, maybeStates, maybeStates, false);
                        
                        // Prepare the right-hand side of the equation system.
                        std::vector<ValueType> b = totalStateRewardVectorGetter(submatrix.getRowCount(), transitionMatrix, maybeStates);

                        // Since we are cutting away target and infinity states, we need to account for this by giving
                        // choices the value infinity that have some successor contained in the infinity states.
                        uint_fast64_t currentRow = 0;
                        for (auto state : maybeStates) {
                            for (uint_fast64_t row = nondeterministicChoiceIndices[state]; row < nondeterministicChoiceIndices[state + 1]; ++row, ++currentRow) {
                                for (auto const& element : transitionMatrix.getRow(row)) {
                                    if (infinityStates.get(element.getColumn())) {
                                        b[currentRow] = storm::utility::infinity<ValueType>();
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Create vector for results for maybe states.
                        std::vector<ValueType> x(maybeStates.getNumberOfSetBits(), storm::utility::zero<ValueType>());
                        
                        // Solve the corresponding system of equations.
                        std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(std::move(submatrix));
                        solver->solveEquations(dir, x, b);
                        
                        // Set values of resulting vector according to result.
                        storm::utility::vector::setVectorValues<ValueType>(result, maybeStates, x);
                    }
                }
                
                // Set values of resulting vector that are known exactly.
                storm::utility::vector::setVectorValues(result, infinityStates, storm::utility::infinity<ValueType>());
                
                return result;
            }
            
            template<typename ValueType>
            std::vector<ValueType> SparseMdpPrctlHelper<ValueType>::computeLongRunAverageProbabilities(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& psiStates, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                // If there are no goal states, we avoid the computation and directly return zero.
                uint_fast64_t numberOfStates = transitionMatrix.getRowGroupCount();
                if (psiStates.empty()) {
                    return std::vector<ValueType>(numberOfStates, storm::utility::zero<ValueType>());
                }
                
                // Likewise, if all bits are set, we can avoid the computation and set.
                if ((~psiStates).empty()) {
                    return std::vector<ValueType>(numberOfStates, storm::utility::one<ValueType>());
                }
                
                // Start by decomposing the MDP into its MECs.
                storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(transitionMatrix, backwardTransitions);
                
                // Get some data members for convenience.
                std::vector<uint_fast64_t> const& nondeterministicChoiceIndices = transitionMatrix.getRowGroupIndices();
                ValueType zero = storm::utility::zero<ValueType>();
                
                //first calculate LRA for the Maximal End Components.
                storm::storage::BitVector statesInMecs(numberOfStates);
                std::vector<uint_fast64_t> stateToMecIndexMap(transitionMatrix.getColumnCount());
                std::vector<ValueType> lraValuesForEndComponents(mecDecomposition.size(), zero);
                
                for (uint_fast64_t currentMecIndex = 0; currentMecIndex < mecDecomposition.size(); ++currentMecIndex) {
                    storm::storage::MaximalEndComponent const& mec = mecDecomposition[currentMecIndex];
                    
                    lraValuesForEndComponents[currentMecIndex] = computeLraForMaximalEndComponent(dir, transitionMatrix, psiStates, mec);
                    
                    // Gather information for later use.
                    for (auto const& stateChoicesPair : mec) {
                        statesInMecs.set(stateChoicesPair.first);
                        stateToMecIndexMap[stateChoicesPair.first] = currentMecIndex;
                    }
                }
                
                // For fast transition rewriting, we build some auxiliary data structures.
                storm::storage::BitVector statesNotContainedInAnyMec = ~statesInMecs;
                uint_fast64_t firstAuxiliaryStateIndex = statesNotContainedInAnyMec.getNumberOfSetBits();
                uint_fast64_t lastStateNotInMecs = 0;
                uint_fast64_t numberOfStatesNotInMecs = 0;
                std::vector<uint_fast64_t> statesNotInMecsBeforeIndex;
                statesNotInMecsBeforeIndex.reserve(numberOfStates);
                for (auto state : statesNotContainedInAnyMec) {
                    while (lastStateNotInMecs <= state) {
                        statesNotInMecsBeforeIndex.push_back(numberOfStatesNotInMecs);
                        ++lastStateNotInMecs;
                    }
                    ++numberOfStatesNotInMecs;
                }
                
                // Finally, we are ready to create the SSP matrix and right-hand side of the SSP.
                std::vector<ValueType> b;
                typename storm::storage::SparseMatrixBuilder<ValueType> sspMatrixBuilder(0, 0, 0, false, true, numberOfStatesNotInMecs + mecDecomposition.size());
                
                // If the source state is not contained in any MEC, we copy its choices (and perform the necessary modifications).
                uint_fast64_t currentChoice = 0;
                for (auto state : statesNotContainedInAnyMec) {
                    sspMatrixBuilder.newRowGroup(currentChoice);
                    
                    for (uint_fast64_t choice = nondeterministicChoiceIndices[state]; choice < nondeterministicChoiceIndices[state + 1]; ++choice, ++currentChoice) {
                        std::vector<ValueType> auxiliaryStateToProbabilityMap(mecDecomposition.size());
                        b.push_back(storm::utility::zero<ValueType>());
                        
                        for (auto element : transitionMatrix.getRow(choice)) {
                            if (statesNotContainedInAnyMec.get(element.getColumn())) {
                                // If the target state is not contained in an MEC, we can copy over the entry.
                                sspMatrixBuilder.addNextValue(currentChoice, statesNotInMecsBeforeIndex[element.getColumn()], element.getValue());
                            } else {
                                // If the target state is contained in MEC i, we need to add the probability to the corresponding field in the vector
                                // so that we are able to write the cumulative probability to the MEC into the matrix.
                                auxiliaryStateToProbabilityMap[stateToMecIndexMap[element.getColumn()]] += element.getValue();
                            }
                        }
                        
                        // Now insert all (cumulative) probability values that target an MEC.
                        for (uint_fast64_t mecIndex = 0; mecIndex < auxiliaryStateToProbabilityMap.size(); ++mecIndex) {
                            if (auxiliaryStateToProbabilityMap[mecIndex] != 0) {
                                sspMatrixBuilder.addNextValue(currentChoice, firstAuxiliaryStateIndex + mecIndex, auxiliaryStateToProbabilityMap[mecIndex]);
                            }
                        }
                    }
                }
                
                // Now we are ready to construct the choices for the auxiliary states.
                for (uint_fast64_t mecIndex = 0; mecIndex < mecDecomposition.size(); ++mecIndex) {
                    storm::storage::MaximalEndComponent const& mec = mecDecomposition[mecIndex];
                    sspMatrixBuilder.newRowGroup(currentChoice);
                    
                    for (auto const& stateChoicesPair : mec) {
                        uint_fast64_t state = stateChoicesPair.first;
                        boost::container::flat_set<uint_fast64_t> const& choicesInMec = stateChoicesPair.second;
                        
                        for (uint_fast64_t choice = nondeterministicChoiceIndices[state]; choice < nondeterministicChoiceIndices[state + 1]; ++choice) {
                            std::vector<ValueType> auxiliaryStateToProbabilityMap(mecDecomposition.size());
                            
                            // If the choice is not contained in the MEC itself, we have to add a similar distribution to the auxiliary state.
                            if (choicesInMec.find(choice) == choicesInMec.end()) {
                                b.push_back(storm::utility::zero<ValueType>());
                                
                                for (auto element : transitionMatrix.getRow(choice)) {
                                    if (statesNotContainedInAnyMec.get(element.getColumn())) {
                                        // If the target state is not contained in an MEC, we can copy over the entry.
                                        sspMatrixBuilder.addNextValue(currentChoice, statesNotInMecsBeforeIndex[element.getColumn()], element.getValue());
                                    } else {
                                        // If the target state is contained in MEC i, we need to add the probability to the corresponding field in the vector
                                        // so that we are able to write the cumulative probability to the MEC into the matrix.
                                        auxiliaryStateToProbabilityMap[stateToMecIndexMap[element.getColumn()]] += element.getValue();
                                    }
                                }
                                
                                // Now insert all (cumulative) probability values that target an MEC.
                                for (uint_fast64_t targetMecIndex = 0; targetMecIndex < auxiliaryStateToProbabilityMap.size(); ++targetMecIndex) {
                                    if (auxiliaryStateToProbabilityMap[targetMecIndex] != 0) {
                                        sspMatrixBuilder.addNextValue(currentChoice, firstAuxiliaryStateIndex + targetMecIndex, auxiliaryStateToProbabilityMap[targetMecIndex]);
                                    }
                                }
                                
                                ++currentChoice;
                            }
                        }
                    }
                    
                    // For each auxiliary state, there is the option to achieve the reward value of the LRA associated with the MEC.
                    ++currentChoice;
                    b.push_back(lraValuesForEndComponents[mecIndex]);
                }
                
                // Finalize the matrix and solve the corresponding system of equations.
                storm::storage::SparseMatrix<ValueType> sspMatrix = sspMatrixBuilder.build(currentChoice);
                
                std::vector<ValueType> sspResult(numberOfStatesNotInMecs + mecDecomposition.size());
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(std::move(sspMatrix));
                solver->solveEquations(dir, sspResult, b);
                
                // Prepare result vector.
                std::vector<ValueType> result(numberOfStates, zero);
                
                // Set the values for states not contained in MECs.
                storm::utility::vector::setVectorValues(result, statesNotContainedInAnyMec, sspResult);
                
                // Set the values for all states in MECs.
                for (auto state : statesInMecs) {
                    result[state] = sspResult[firstAuxiliaryStateIndex + stateToMecIndexMap[state]];
                }
                
                return result;
            }
            
            template<typename ValueType>
            ValueType SparseMdpPrctlHelper<ValueType>::computeLraForMaximalEndComponent(OptimizationDirection dir, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const& psiStates, storm::storage::MaximalEndComponent const& mec) {
                std::shared_ptr<storm::solver::LpSolver> solver = storm::utility::solver::getLpSolver("LRA for MEC");
                solver->setOptimizationDirection(invert(dir));
                
                // First, we need to create the variables for the problem.
                std::map<uint_fast64_t, storm::expressions::Variable> stateToVariableMap;
                for (auto const& stateChoicesPair : mec) {
                    std::string variableName = "h" + std::to_string(stateChoicesPair.first);
                    stateToVariableMap[stateChoicesPair.first] = solver->addUnboundedContinuousVariable(variableName);
                }
                storm::expressions::Variable lambda = solver->addUnboundedContinuousVariable("L", 1);
                solver->update();
                
                // Now we encode the problem as constraints.
                for (auto const& stateChoicesPair : mec) {
                    uint_fast64_t state = stateChoicesPair.first;
                    
                    // Now, based on the type of the state, create a suitable constraint.
                    for (auto choice : stateChoicesPair.second) {
                        storm::expressions::Expression constraint = -lambda;
                        ValueType r = 0;
                        
                        for (auto element : transitionMatrix.getRow(choice)) {
                            constraint = constraint + stateToVariableMap.at(element.getColumn()) * solver->getConstant(storm::utility::convertNumber<double>(element.getValue()));
                            if (psiStates.get(element.getColumn())) {
                                r += element.getValue();
                            }
                        }
                        constraint = solver->getConstant(storm::utility::convertNumber<double>(r)) + constraint;
                        
                        if (dir == OptimizationDirection::Minimize) {
                            constraint = stateToVariableMap.at(state) <= constraint;
                        } else {
                            constraint = stateToVariableMap.at(state) >= constraint;
                        }
                        solver->addConstraint("state" + std::to_string(state) + "," + std::to_string(choice), constraint);
                    }
                }
                
                solver->optimize();
                return storm::utility::convertNumber<ValueType>(solver->getContinuousValue(lambda));
            }
            
            template<typename ValueType>
            std::unique_ptr<CheckResult> SparseMdpPrctlHelper<ValueType>::computeConditionalProbabilities(OptimizationDirection dir, storm::storage::sparse::state_type initialState, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& targetStates, storm::storage::BitVector const& conditionStates, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                
                // For the max-case, we can simply take the given target states. For the min-case, however, we need to
                // find the MECs of non-target states and make them the new target states.
                storm::storage::BitVector fixedTargetStates;
                if (dir == OptimizationDirection::Maximize) {
                    fixedTargetStates = targetStates;
                } else {
                    fixedTargetStates = storm::storage::BitVector(targetStates.size());
                    storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(transitionMatrix, backwardTransitions, ~targetStates);
                    for (auto const& mec : mecDecomposition) {
                        for (auto const& stateActionsPair : mec) {
                            fixedTargetStates.set(stateActionsPair.first);
                        }
                    }
                }
                
                // We solve the max-case and later adjust the result if the optimization direction was to minimize.
                storm::storage::BitVector initialStatesBitVector(transitionMatrix.getRowGroupCount());
                initialStatesBitVector.set(initialState);
                
                storm::storage::BitVector allStates(initialStatesBitVector.size(), true);
                std::vector<ValueType> conditionProbabilities = std::move(computeUntilProbabilities(OptimizationDirection::Maximize, transitionMatrix, backwardTransitions, allStates, conditionStates, false, false, minMaxLinearEquationSolverFactory).values);
                
                // If the conditional probability is undefined for the initial state, we return directly.
                if (storm::utility::isZero(conditionProbabilities[initialState])) {
                    return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(initialState, storm::utility::infinity<ValueType>()));
                }
                
                std::vector<ValueType> targetProbabilities = std::move(computeUntilProbabilities(OptimizationDirection::Maximize, transitionMatrix, backwardTransitions, allStates, fixedTargetStates, false, false, minMaxLinearEquationSolverFactory).values);

                storm::storage::BitVector statesWithProbabilityGreater0E(transitionMatrix.getRowGroupCount(), true);
                storm::storage::sparse::state_type state = 0;
                for (auto const& element : conditionProbabilities) {
                    if (storm::utility::isZero(element)) {
                        statesWithProbabilityGreater0E.set(state, false);
                    }
                    ++state;
                }

                // Determine those states that need to be equipped with a restart mechanism.
                storm::storage::BitVector problematicStates = storm::utility::graph::performProb0E(transitionMatrix, transitionMatrix.getRowGroupIndices(), backwardTransitions, allStates, conditionStates | fixedTargetStates);

                // Otherwise, we build the transformed MDP.
                storm::storage::BitVector relevantStates = storm::utility::graph::getReachableStates(transitionMatrix, initialStatesBitVector, allStates, conditionStates | fixedTargetStates);
                std::vector<uint_fast64_t> numberOfStatesBeforeRelevantStates = relevantStates.getNumberOfSetBitsBeforeIndices();
                storm::storage::sparse::state_type newGoalState = relevantStates.getNumberOfSetBits();
                storm::storage::sparse::state_type newStopState = newGoalState + 1;
                storm::storage::sparse::state_type newFailState = newStopState + 1;
                
                // Build the transitions of the (relevant) states of the original model.
                storm::storage::SparseMatrixBuilder<ValueType> builder(0, newFailState + 1, 0, true, true);
                uint_fast64_t currentRow = 0;
                for (auto state : relevantStates) {
                    builder.newRowGroup(currentRow);
                    if (fixedTargetStates.get(state)) {
                        builder.addNextValue(currentRow, newGoalState, conditionProbabilities[state]);
                        if (!storm::utility::isZero(conditionProbabilities[state])) {
                            builder.addNextValue(currentRow, newFailState, storm::utility::one<ValueType>() - conditionProbabilities[state]);
                        }
                        ++currentRow;
                    } else if (conditionStates.get(state)) {
                        builder.addNextValue(currentRow, newGoalState, targetProbabilities[state]);
                        if (!storm::utility::isZero(targetProbabilities[state])) {
                            builder.addNextValue(currentRow, newStopState, storm::utility::one<ValueType>() - targetProbabilities[state]);
                        }
                        ++currentRow;
                    } else {
                        for (uint_fast64_t row = transitionMatrix.getRowGroupIndices()[state]; row < transitionMatrix.getRowGroupIndices()[state + 1]; ++row) {
                            for (auto const& successorEntry : transitionMatrix.getRow(row)) {
                                builder.addNextValue(currentRow, numberOfStatesBeforeRelevantStates[successorEntry.getColumn()], successorEntry.getValue());
                            }
                            ++currentRow;
                        }
                        if (problematicStates.get(state)) {
                            builder.addNextValue(currentRow, numberOfStatesBeforeRelevantStates[initialState], storm::utility::one<ValueType>());
                            ++currentRow;
                        }
                    }
                }
                
                // Now build the transitions of the newly introduced states.
                builder.newRowGroup(currentRow);
                builder.addNextValue(currentRow, newGoalState, storm::utility::one<ValueType>());
                ++currentRow;
                builder.newRowGroup(currentRow);
                builder.addNextValue(currentRow, newStopState, storm::utility::one<ValueType>());
                ++currentRow;
                builder.newRowGroup(currentRow);
                builder.addNextValue(currentRow, numberOfStatesBeforeRelevantStates[initialState], storm::utility::one<ValueType>());
                ++currentRow;
                
                // Finally, build the matrix and dispatch the query as a reachability query.
                storm::storage::BitVector newGoalStates(newFailState + 1);
                newGoalStates.set(newGoalState);
                storm::storage::SparseMatrix<ValueType> newTransitionMatrix = builder.build();
                storm::storage::SparseMatrix<ValueType> newBackwardTransitions = newTransitionMatrix.transpose(true);
                std::vector<ValueType> goalProbabilities = std::move(computeUntilProbabilities(OptimizationDirection::Maximize, newTransitionMatrix, newBackwardTransitions, storm::storage::BitVector(newFailState + 1, true), newGoalStates, false, false, minMaxLinearEquationSolverFactory).values);
                
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(initialState, dir == OptimizationDirection::Maximize ? goalProbabilities[numberOfStatesBeforeRelevantStates[initialState]] : storm::utility::one<ValueType>() - goalProbabilities[numberOfStatesBeforeRelevantStates[initialState]]));
            }
            
            template class SparseMdpPrctlHelper<double>;
            template std::vector<double> SparseMdpPrctlHelper<double>::computeInstantaneousRewards(OptimizationDirection dir, storm::storage::SparseMatrix<double> const& transitionMatrix, storm::models::sparse::StandardRewardModel<double> const& rewardModel, uint_fast64_t stepCount, storm::solver::MinMaxLinearEquationSolverFactory<double> const& minMaxLinearEquationSolverFactory);
            template std::vector<double> SparseMdpPrctlHelper<double>::computeCumulativeRewards(OptimizationDirection dir, storm::storage::SparseMatrix<double> const& transitionMatrix, storm::models::sparse::StandardRewardModel<double> const& rewardModel, uint_fast64_t stepBound, storm::solver::MinMaxLinearEquationSolverFactory<double> const& minMaxLinearEquationSolverFactory);
            template std::vector<double> SparseMdpPrctlHelper<double>::computeReachabilityRewards(OptimizationDirection dir, storm::storage::SparseMatrix<double> const& transitionMatrix, storm::storage::SparseMatrix<double> const& backwardTransitions, storm::models::sparse::StandardRewardModel<double> const& rewardModel, storm::storage::BitVector const& targetStates, bool qualitative, storm::solver::MinMaxLinearEquationSolverFactory<double> const& minMaxLinearEquationSolverFactory);

#ifdef STORM_HAVE_CARL
            template class SparseMdpPrctlHelper<storm::RationalNumber>;
            template std::vector<storm::RationalNumber> SparseMdpPrctlHelper<storm::RationalNumber>::computeInstantaneousRewards(OptimizationDirection dir, storm::storage::SparseMatrix<storm::RationalNumber> const& transitionMatrix, storm::models::sparse::StandardRewardModel<storm::RationalNumber> const& rewardModel, uint_fast64_t stepCount, storm::solver::MinMaxLinearEquationSolverFactory<storm::RationalNumber> const& minMaxLinearEquationSolverFactory);
            template std::vector<storm::RationalNumber> SparseMdpPrctlHelper<storm::RationalNumber>::computeCumulativeRewards(OptimizationDirection dir, storm::storage::SparseMatrix<storm::RationalNumber> const& transitionMatrix, storm::models::sparse::StandardRewardModel<storm::RationalNumber> const& rewardModel, uint_fast64_t stepBound, storm::solver::MinMaxLinearEquationSolverFactory<storm::RationalNumber> const& minMaxLinearEquationSolverFactory);
            template std::vector<storm::RationalNumber> SparseMdpPrctlHelper<storm::RationalNumber>::computeReachabilityRewards(OptimizationDirection dir, storm::storage::SparseMatrix<storm::RationalNumber> const& transitionMatrix, storm::storage::SparseMatrix<storm::RationalNumber> const& backwardTransitions, storm::models::sparse::StandardRewardModel<storm::RationalNumber> const& rewardModel, storm::storage::BitVector const& targetStates, bool qualitative, storm::solver::MinMaxLinearEquationSolverFactory<storm::RationalNumber> const& minMaxLinearEquationSolverFactory);
#endif
        }
    }
}
