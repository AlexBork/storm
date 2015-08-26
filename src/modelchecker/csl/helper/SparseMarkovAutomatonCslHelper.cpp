#include "src/modelchecker/csl/helper/SparseMarkovAutomatonCslHelper.h"

#include "src/modelchecker/prctl/helper/SparseMdpPrctlHelper.h"

#include "src/storage/StronglyConnectedComponentDecomposition.h"
#include "src/storage/MaximalEndComponentDecomposition.h"

#include "src/settings/SettingsManager.h"
#include "src/settings/modules/GeneralSettings.h"

#include "src/utility/macros.h"
#include "src/utility/vector.h"
#include "src/utility/graph.h"

#include "src/storage/expressions/Variable.h"
#include "src/storage/expressions/Expression.h"

#include "src/utility/numerical.h"

#include "src/solver/MinMaxLinearEquationSolver.h"
#include "src/solver/LpSolver.h"

#include "src/exceptions/InvalidStateException.h"
#include "src/exceptions/InvalidPropertyException.h"

namespace storm {
    namespace modelchecker {
        namespace helper {
            template<typename ValueType, typename RewardModelType>
            void SparseMarkovAutomatonCslHelper<ValueType, RewardModelType>::computeBoundedReachabilityProbabilities(bool min, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& exitRates, storm::storage::BitVector const& markovianStates, storm::storage::BitVector const& goalStates, storm::storage::BitVector const& markovianNonGoalStates, storm::storage::BitVector const& probabilisticNonGoalStates, std::vector<ValueType>& markovianNonGoalValues, std::vector<ValueType>& probabilisticNonGoalValues, ValueType delta, uint_fast64_t numberOfSteps, storm::utility::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                // Start by computing four sparse matrices:
                // * a matrix aMarkovian with all (discretized) transitions from Markovian non-goal states to all Markovian non-goal states.
                // * a matrix aMarkovianToProbabilistic with all (discretized) transitions from Markovian non-goal states to all probabilistic non-goal states.
                // * a matrix aProbabilistic with all (non-discretized) transitions from probabilistic non-goal states to other probabilistic non-goal states.
                // * a matrix aProbabilisticToMarkovian with all (non-discretized) transitions from probabilistic non-goal states to all Markovian non-goal states.
                typename storm::storage::SparseMatrix<ValueType> aMarkovian = transitionMatrix.getSubmatrix(true, markovianNonGoalStates, markovianNonGoalStates, true);
                typename storm::storage::SparseMatrix<ValueType> aMarkovianToProbabilistic = transitionMatrix.getSubmatrix(true, markovianNonGoalStates, probabilisticNonGoalStates);
                typename storm::storage::SparseMatrix<ValueType> aProbabilistic = transitionMatrix.getSubmatrix(true, probabilisticNonGoalStates, probabilisticNonGoalStates);
                typename storm::storage::SparseMatrix<ValueType> aProbabilisticToMarkovian = transitionMatrix.getSubmatrix(true, probabilisticNonGoalStates, markovianNonGoalStates);
                
                // The matrices with transitions from Markovian states need to be digitized.
                // Digitize aMarkovian. Based on whether the transition is a self-loop or not, we apply the two digitization rules.
                uint_fast64_t rowIndex = 0;
                for (auto state : markovianNonGoalStates) {
                    for (auto& element : aMarkovian.getRow(rowIndex)) {
                        ValueType eTerm = std::exp(-exitRates[state] * delta);
                        if (element.getColumn() == rowIndex) {
                            element.setValue((storm::utility::one<ValueType>() - eTerm) * element.getValue() + eTerm);
                        } else {
                            element.setValue((storm::utility::one<ValueType>() - eTerm) * element.getValue());
                        }
                    }
                    ++rowIndex;
                }
                
                // Digitize aMarkovianToProbabilistic. As there are no self-loops in this case, we only need to apply the digitization formula for regular successors.
                rowIndex = 0;
                for (auto state : markovianNonGoalStates) {
                    for (auto& element : aMarkovianToProbabilistic.getRow(rowIndex)) {
                        element.setValue((1 - std::exp(-exitRates[state] * delta)) * element.getValue());
                    }
                    ++rowIndex;
                }
                
                // Initialize the two vectors that hold the variable one-step probabilities to all target states for probabilistic and Markovian (non-goal) states.
                std::vector<ValueType> bProbabilistic(aProbabilistic.getRowCount());
                std::vector<ValueType> bMarkovian(markovianNonGoalStates.getNumberOfSetBits());
                
                // Compute the two fixed right-hand side vectors, one for Markovian states and one for the probabilistic ones.
                std::vector<ValueType> bProbabilisticFixed = transitionMatrix.getConstrainedRowSumVector(probabilisticNonGoalStates, goalStates);
                std::vector<ValueType> bMarkovianFixed;
                bMarkovianFixed.reserve(markovianNonGoalStates.getNumberOfSetBits());
                for (auto state : markovianNonGoalStates) {
                    bMarkovianFixed.push_back(storm::utility::zero<ValueType>());
                    
                    for (auto& element : transitionMatrix.getRowGroup(state)) {
                        if (goalStates.get(element.getColumn())) {
                            bMarkovianFixed.back() += (1 - std::exp(-exitRates[state] * delta)) * element.getValue();
                        }
                    }
                }
                
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(aProbabilistic);
                
                // Perform the actual value iteration
                // * loop until the step bound has been reached
                // * in the loop:
                // *    perform value iteration using A_PSwG, v_PS and the vector b where b = (A * 1_G)|PS + A_PStoMS * v_MS
                //      and 1_G being the characteristic vector for all goal states.
                // *    perform one timed-step using v_MS := A_MSwG * v_MS + A_MStoPS * v_PS + (A * 1_G)|MS
                std::vector<ValueType> markovianNonGoalValuesSwap(markovianNonGoalValues);
                std::vector<ValueType> multiplicationResultScratchMemory(aProbabilistic.getRowCount());
                std::vector<ValueType> aProbabilisticScratchMemory(probabilisticNonGoalValues.size());
                for (uint_fast64_t currentStep = 0; currentStep < numberOfSteps; ++currentStep) {
                    // Start by (re-)computing bProbabilistic = bProbabilisticFixed + aProbabilisticToMarkovian * vMarkovian.
                    aProbabilisticToMarkovian.multiplyWithVector(markovianNonGoalValues, bProbabilistic);
                    storm::utility::vector::addVectors(bProbabilistic, bProbabilisticFixed, bProbabilistic);
                    
                    // Now perform the inner value iteration for probabilistic states.
                    solver->solveEquationSystem(min, probabilisticNonGoalValues, bProbabilistic, &multiplicationResultScratchMemory, &aProbabilisticScratchMemory);
                    
                    // (Re-)compute bMarkovian = bMarkovianFixed + aMarkovianToProbabilistic * vProbabilistic.
                    aMarkovianToProbabilistic.multiplyWithVector(probabilisticNonGoalValues, bMarkovian);
                    storm::utility::vector::addVectors(bMarkovian, bMarkovianFixed, bMarkovian);
                    aMarkovian.multiplyWithVector(markovianNonGoalValues, markovianNonGoalValuesSwap);
                    std::swap(markovianNonGoalValues, markovianNonGoalValuesSwap);
                    storm::utility::vector::addVectors(markovianNonGoalValues, bMarkovian, markovianNonGoalValues);
                }
                
                // After the loop, perform one more step of the value iteration for PS states.
                aProbabilisticToMarkovian.multiplyWithVector(markovianNonGoalValues, bProbabilistic);
                storm::utility::vector::addVectors(bProbabilistic, bProbabilisticFixed, bProbabilistic);
                solver->solveEquationSystem(min, probabilisticNonGoalValues, bProbabilistic, &multiplicationResultScratchMemory, &aProbabilisticScratchMemory);
            }
            
            template<typename ValueType, typename RewardModelType>
            std::vector<ValueType> SparseMarkovAutomatonCslHelper<ValueType, RewardModelType>::computeBoundedUntilProbabilities(bool minimize, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& markovianStates, storm::storage::BitVector const& psiStates, std::pair<double, double> const& boundsPair, storm::utility::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                uint_fast64_t numberOfStates = transitionMatrix.getRowGroupCount();
                
                // 'Unpack' the bounds to make them more easily accessible.
                double lowerBound = boundsPair.first;
                double upperBound = boundsPair.second;
                
                // (1) Compute the accuracy we need to achieve the required error bound.
                ValueType maxExitRate = 0;
                for (auto value : exitRateVector) {
                    maxExitRate = std::max(maxExitRate, value);
                }
                ValueType delta = (2 * storm::settings::generalSettings().getPrecision()) / (upperBound * maxExitRate * maxExitRate);
                
                // (2) Compute the number of steps we need to make for the interval.
                uint_fast64_t numberOfSteps = static_cast<uint_fast64_t>(std::ceil((upperBound - lowerBound) / delta));
                STORM_LOG_INFO("Performing " << numberOfSteps << " iterations (delta=" << delta << ") for interval [" << lowerBound << ", " << upperBound << "]." << std::endl);
                
                // (3) Compute the non-goal states and initialize two vectors
                // * vProbabilistic holds the probability values of probabilistic non-goal states.
                // * vMarkovian holds the probability values of Markovian non-goal states.
                storm::storage::BitVector const& markovianNonGoalStates = markovianStates & ~psiStates;
                storm::storage::BitVector const& probabilisticNonGoalStates = ~markovianStates & ~psiStates;
                std::vector<ValueType> vProbabilistic(probabilisticNonGoalStates.getNumberOfSetBits());
                std::vector<ValueType> vMarkovian(markovianNonGoalStates.getNumberOfSetBits());
                
                computeBoundedReachabilityProbabilities(minimize, transitionMatrix, exitRateVector, markovianStates, psiStates, markovianNonGoalStates, probabilisticNonGoalStates, vMarkovian, vProbabilistic, delta, numberOfSteps, minMaxLinearEquationSolverFactory);
                
                // (4) If the lower bound of interval was non-zero, we need to take the current values as the starting values for a subsequent value iteration.
                if (lowerBound != storm::utility::zero<ValueType>()) {
                    std::vector<ValueType> vAllProbabilistic((~markovianStates).getNumberOfSetBits());
                    std::vector<ValueType> vAllMarkovian(markovianStates.getNumberOfSetBits());
                    
                    // Create the starting value vectors for the next value iteration based on the results of the previous one.
                    storm::utility::vector::setVectorValues<ValueType>(vAllProbabilistic, psiStates % ~markovianStates, storm::utility::one<ValueType>());
                    storm::utility::vector::setVectorValues<ValueType>(vAllProbabilistic, ~psiStates % ~markovianStates, vProbabilistic);
                    storm::utility::vector::setVectorValues<ValueType>(vAllMarkovian, psiStates % markovianStates, storm::utility::one<ValueType>());
                    storm::utility::vector::setVectorValues<ValueType>(vAllMarkovian, ~psiStates % markovianStates, vMarkovian);
                    
                    // Compute the number of steps to reach the target interval.
                    numberOfSteps = static_cast<uint_fast64_t>(std::ceil(lowerBound / delta));
                    STORM_LOG_INFO("Performing " << numberOfSteps << " iterations (delta=" << delta << ") for interval [0, " << lowerBound << "]." << std::endl);
                    
                    // Compute the bounded reachability for interval [0, b-a].
                    computeBoundedReachabilityProbabilities(minimize, transitionMatrix, exitRateVector, markovianStates, storm::storage::BitVector(numberOfStates), markovianStates, ~markovianStates, vAllMarkovian, vAllProbabilistic, delta, numberOfSteps, minMaxLinearEquationSolverFactory);
                    
                    // Create the result vector out of vAllProbabilistic and vAllMarkovian and return it.
                    std::vector<ValueType> result(numberOfStates, storm::utility::zero<ValueType>());
                    storm::utility::vector::setVectorValues(result, ~markovianStates, vAllProbabilistic);
                    storm::utility::vector::setVectorValues(result, markovianStates, vAllMarkovian);
                    
                    return result;
                } else {
                    // Create the result vector out of 1_G, vProbabilistic and vMarkovian and return it.
                    std::vector<ValueType> result(numberOfStates);
                    storm::utility::vector::setVectorValues<ValueType>(result, psiStates, storm::utility::one<ValueType>());
                    storm::utility::vector::setVectorValues(result, probabilisticNonGoalStates, vProbabilistic);
                    storm::utility::vector::setVectorValues(result, markovianNonGoalStates, vMarkovian);
                    return result;
                }
            }
            
            template<typename ValueType, typename RewardModelType>
            std::vector<ValueType> SparseMarkovAutomatonCslHelper<ValueType, RewardModelType>::computeUntilProbabilities(bool minimize, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::utility::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                return storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType, RewardModelType>::computeUntilProbabilities(minimize, transitionMatrix, backwardTransitions, phiStates, psiStates, qualitative, minMaxLinearEquationSolverFactory);
            }
            
            template<typename ValueType, typename RewardModelType>
            std::vector<ValueType> SparseMarkovAutomatonCslHelper<ValueType, RewardModelType>::computeReachabilityRewards(bool minimize, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& markovianStates, RewardModelType const& rewardModel, storm::storage::BitVector const& psiStates, bool qualitative, storm::utility::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                std::vector<ValueType> totalRewardVector = rewardModel.getTotalRewardVector(transitionMatrix.getRowCount(), transitionMatrix, storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true));
                return computeExpectedRewards(minimize, transitionMatrix, backwardTransitions, exitRateVector, markovianStates, psiStates, totalRewardVector, minMaxLinearEquationSolverFactory);
            }
            
            template<typename ValueType, typename RewardModelType>
            std::vector<ValueType> SparseMarkovAutomatonCslHelper<ValueType, RewardModelType>::computeLongRunAverage(bool minimize, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& markovianStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::utility::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                uint_fast64_t numberOfStates = transitionMatrix.getRowGroupCount();
                
                // If there are no goal states, we avoid the computation and directly return zero.
                if (psiStates.empty()) {
                    return std::vector<ValueType>(numberOfStates, storm::utility::zero<ValueType>());
                }
                
                // Likewise, if all bits are set, we can avoid the computation and set.
                if ((~psiStates).empty()) {
                    return std::vector<ValueType>(numberOfStates, storm::utility::one<ValueType>());
                }
                
                // Start by decomposing the Markov automaton into its MECs.
                storm::storage::MaximalEndComponentDecomposition<double> mecDecomposition(transitionMatrix, backwardTransitions);
                
                // Get some data members for convenience.
                std::vector<uint_fast64_t> const& nondeterministicChoiceIndices = transitionMatrix.getRowGroupIndices();
                
                // Now start with compute the long-run average for all end components in isolation.
                std::vector<ValueType> lraValuesForEndComponents;
                
                // While doing so, we already gather some information for the following steps.
                std::vector<uint_fast64_t> stateToMecIndexMap(numberOfStates);
                storm::storage::BitVector statesInMecs(numberOfStates);
                
                for (uint_fast64_t currentMecIndex = 0; currentMecIndex < mecDecomposition.size(); ++currentMecIndex) {
                    storm::storage::MaximalEndComponent const& mec = mecDecomposition[currentMecIndex];
                    
                    // Gather information for later use.
                    for (auto const& stateChoicesPair : mec) {
                        uint_fast64_t state = stateChoicesPair.first;
                        
                        statesInMecs.set(state);
                        stateToMecIndexMap[state] = currentMecIndex;
                    }
                    
                    // Compute the LRA value for the current MEC.
                    lraValuesForEndComponents.push_back(computeLraForMaximalEndComponent(minimize, transitionMatrix, exitRateVector, markovianStates, psiStates, mec));
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
                
                std::vector<ValueType> x(numberOfStatesNotInMecs + mecDecomposition.size());
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(sspMatrix);
                solver->solveEquationSystem(minimize, x, b);
                
                // Prepare result vector.
                std::vector<ValueType> result(numberOfStates);
                
                // Set the values for states not contained in MECs.
                storm::utility::vector::setVectorValues(result, statesNotContainedInAnyMec, x);
                
                // Set the values for all states in MECs.
                for (auto state : statesInMecs) {
                    result[state] = x[firstAuxiliaryStateIndex + stateToMecIndexMap[state]];
                }
                
                return result;
            }
            
            template<typename ValueType, typename RewardModelType>
            std::vector<ValueType> SparseMarkovAutomatonCslHelper<ValueType, RewardModelType>::computeExpectedTimes(bool minimize, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& markovianStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::utility::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                uint_fast64_t numberOfStates = transitionMatrix.getRowGroupCount();
                std::vector<ValueType> rewardValues(numberOfStates, storm::utility::zero<ValueType>());
                storm::utility::vector::setVectorValues(rewardValues, markovianStates, storm::utility::one<ValueType>());
                return computeExpectedRewards(minimize, transitionMatrix, backwardTransitions, exitRateVector, markovianStates, psiStates, rewardValues, minMaxLinearEquationSolverFactory);
            }
            
            template<typename ValueType, typename RewardModelType>
            std::vector<ValueType> SparseMarkovAutomatonCslHelper<ValueType, RewardModelType>::computeExpectedRewards(bool minimize, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& markovianStates, storm::storage::BitVector const& goalStates, std::vector<ValueType> const& stateRewards, storm::utility::solver::MinMaxLinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory) {
                uint_fast64_t numberOfStates = transitionMatrix.getRowGroupCount();
                
                // First, we need to check which states have infinite expected time (by definition).
                storm::storage::BitVector infinityStates;
                if (minimize) {
                    // If we need to compute the minimum expected times, we have to set the values of those states to infinity that, under all schedulers,
                    // reach a bottom SCC without a goal state.
                    
                    // So we start by computing all bottom SCCs without goal states.
                    storm::storage::StronglyConnectedComponentDecomposition<double> sccDecomposition(transitionMatrix, ~goalStates, true, true);
                    
                    // Now form the union of all these SCCs.
                    storm::storage::BitVector unionOfNonGoalBSccs(numberOfStates);
                    for (auto const& scc : sccDecomposition) {
                        for (auto state : scc) {
                            unionOfNonGoalBSccs.set(state);
                        }
                    }
                    
                    // Finally, if this union is non-empty, compute the states such that all schedulers reach some state of the union.
                    if (!unionOfNonGoalBSccs.empty()) {
                        infinityStates = storm::utility::graph::performProbGreater0A(transitionMatrix, transitionMatrix.getRowGroupIndices(), backwardTransitions, storm::storage::BitVector(numberOfStates, true), unionOfNonGoalBSccs);
                    } else {
                        // Otherwise, we have no infinity states.
                        infinityStates = storm::storage::BitVector(numberOfStates);
                    }
                } else {
                    // If we maximize the property, the expected time of a state is infinite, if an end-component without any goal state is reachable.
                    
                    // So we start by computing all MECs that have no goal state.
                    storm::storage::MaximalEndComponentDecomposition<double> mecDecomposition(transitionMatrix, backwardTransitions, ~goalStates);
                    
                    // Now we form the union of all states in these end components.
                    storm::storage::BitVector unionOfNonGoalMaximalEndComponents(numberOfStates);
                    for (auto const& mec : mecDecomposition) {
                        for (auto const& stateActionPair : mec) {
                            unionOfNonGoalMaximalEndComponents.set(stateActionPair.first);
                        }
                    }
                    
                    if (!unionOfNonGoalMaximalEndComponents.empty()) {
                        // Now we need to check for which states there exists a scheduler that reaches one of the previously computed states.
                        infinityStates = storm::utility::graph::performProbGreater0E(transitionMatrix, transitionMatrix.getRowGroupIndices(), backwardTransitions, storm::storage::BitVector(numberOfStates, true), unionOfNonGoalMaximalEndComponents);
                    } else {
                        // Otherwise, we have no infinity states.
                        infinityStates = storm::storage::BitVector(numberOfStates);
                    }
                }
                
                // Now we identify the states for which values need to be computed.
                storm::storage::BitVector maybeStates = ~(goalStates | infinityStates);
                
                // Then, we can eliminate the rows and columns for all states whose values are already known to be 0.
                std::vector<ValueType> x(maybeStates.getNumberOfSetBits());
                storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, maybeStates, maybeStates);
                
                // Now prepare the expected reward values for all states so they can be used as the right-hand side of the equation system.
                std::vector<ValueType> rewardValues(stateRewards);
                for (auto state : markovianStates) {
                    rewardValues[state] = rewardValues[state] / exitRateVector[state];
                }
                
                // Finally, prepare the actual right-hand side.
                std::vector<ValueType> b(submatrix.getRowCount());
                storm::utility::vector::selectVectorValuesRepeatedly(b, maybeStates, transitionMatrix.getRowGroupIndices(), rewardValues);
                
                // Solve the corresponding system of equations.
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(submatrix);
                solver->solveEquationSystem(minimize, x, b);
                
                // Create resulting vector.
                std::vector<ValueType> result(numberOfStates);
                
                // Set values of resulting vector according to previous result and return the result.
                storm::utility::vector::setVectorValues<ValueType>(result, maybeStates, x);
                storm::utility::vector::setVectorValues(result, goalStates, storm::utility::zero<ValueType>());
                storm::utility::vector::setVectorValues(result, infinityStates, storm::utility::infinity<ValueType>());
                
                return result;
            }
            
            template<typename ValueType, typename RewardModelType>
            ValueType SparseMarkovAutomatonCslHelper<ValueType, RewardModelType>::computeLraForMaximalEndComponent(bool minimize, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& markovianStates, storm::storage::BitVector const& goalStates, storm::storage::MaximalEndComponent const& mec) {
                std::unique_ptr<storm::utility::solver::LpSolverFactory> lpSolverFactory(new storm::utility::solver::LpSolverFactory());
                std::unique_ptr<storm::solver::LpSolver> solver = lpSolverFactory->create("LRA for MEC");
                solver->setModelSense(minimize ? storm::solver::LpSolver::ModelSense::Maximize : storm::solver::LpSolver::ModelSense::Minimize);
                
                // First, we need to create the variables for the problem.
                std::map<uint_fast64_t, storm::expressions::Variable> stateToVariableMap;
                for (auto const& stateChoicesPair : mec) {
                    std::string variableName = "x" + std::to_string(stateChoicesPair.first);
                    stateToVariableMap[stateChoicesPair.first] = solver->addUnboundedContinuousVariable(variableName);
                }
                storm::expressions::Variable k = solver->addUnboundedContinuousVariable("k", 1);
                solver->update();
                
                // Now we encode the problem as constraints.
                std::vector<uint_fast64_t> const& nondeterministicChoiceIndices = transitionMatrix.getRowGroupIndices();
                for (auto const& stateChoicesPair : mec) {
                    uint_fast64_t state = stateChoicesPair.first;
                    
                    // Now, based on the type of the state, create a suitable constraint.
                    if (markovianStates.get(state)) {
                        storm::expressions::Expression constraint = stateToVariableMap.at(state);
                        
                        for (auto element : transitionMatrix.getRow(nondeterministicChoiceIndices[state])) {
                            constraint = constraint - stateToVariableMap.at(element.getColumn()) * solver->getConstant(element.getValue());
                        }
                        
                        constraint = constraint + solver->getConstant(storm::utility::one<ValueType>() / exitRateVector[state]) * k;
                        storm::expressions::Expression rightHandSide = goalStates.get(state) ? solver->getConstant(storm::utility::one<ValueType>() / exitRateVector[state]) : solver->getConstant(0);
                        if (minimize) {
                            constraint = constraint <= rightHandSide;
                        } else {
                            constraint = constraint >= rightHandSide;
                        }
                        solver->addConstraint("state" + std::to_string(state), constraint);
                    } else {
                        // For probabilistic states, we want to add the constraint x_s <= sum P(s, a, s') * x_s' where a is the current action
                        // and the sum ranges over all states s'.
                        for (auto choice : stateChoicesPair.second) {
                            storm::expressions::Expression constraint = stateToVariableMap.at(state);
                            
                            for (auto element : transitionMatrix.getRow(choice)) {
                                constraint = constraint - stateToVariableMap.at(element.getColumn()) * solver->getConstant(element.getValue());
                            }
                            
                            storm::expressions::Expression rightHandSide = solver->getConstant(storm::utility::zero<ValueType>());
                            if (minimize) {
                                constraint = constraint <= rightHandSide;
                            } else {
                                constraint = constraint >= rightHandSide;
                            }
                            solver->addConstraint("state" + std::to_string(state), constraint);
                        }
                    }
                }
                
                solver->optimize();
                return solver->getContinuousValue(k);
            }
            
            template class SparseMarkovAutomatonCslHelper<double>;
            
        }
    }
}