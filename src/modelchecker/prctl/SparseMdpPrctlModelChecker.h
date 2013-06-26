/*
 * SparseMdpPrctlModelChecker.h
 *
 *  Created on: 15.02.2013
 *      Author: Christian Dehnert
 */

#ifndef STORM_MODELCHECKER_PRCTL_SPARSEMDPPRCTLMODELCHECKER_H_
#define STORM_MODELCHECKER_PRCTL_SPARSEMDPPRCTLMODELCHECKER_H_

#include <vector>
#include <stack>
#include <fstream>

#include "src/modelchecker/prctl/AbstractModelChecker.h"
#include "src/solver/AbstractNondeterministicLinearEquationSolver.h"
#include "src/solver/GmmxxLinearEquationSolver.h"
#include "src/models/Mdp.h"
#include "src/utility/vector.h"
#include "src/utility/graph.h"
#include "src/utility/Settings.h"

namespace storm {
    namespace modelchecker {
        namespace prctl {
            
            /*!
             * This class represents the base class for all PRCTL model checkers for MDPs.
             */
            template<class Type>
            class SparseMdpPrctlModelChecker : public AbstractModelChecker<Type> {
                
            public:
                /*!
                 * Constructs a SparseMdpPrctlModelChecker with the given model.
                 *
                 * @param model The MDP to be checked.
                 */
                explicit SparseMdpPrctlModelChecker(storm::models::Mdp<Type> const& model, storm::solver::AbstractNondeterministicLinearEquationSolver<Type>* linearEquationSolver) : AbstractModelChecker<Type>(model), minimumOperatorStack(), linearEquationSolver(linearEquationSolver) {
                    // Intentionally left empty.
                }
                
                /*!
                 * Copy constructs a SparseMdpPrctlModelChecker from the given model checker. In particular, this means that the newly
                 * constructed model checker will have the model of the given model checker as its associated model.
                 */
                explicit SparseMdpPrctlModelChecker(storm::modelchecker::prctl::SparseMdpPrctlModelChecker<Type> const& modelchecker)
                : AbstractModelChecker<Type>(modelchecker),  minimumOperatorStack(), linearEquationSolver(new storm::solver::AbstractNondeterministicLinearEquationSolver<Type>()) {
                    // Intentionally left empty.
                }
                
                /*!
                 * Returns a constant reference to the MDP associated with this model checker.
                 * @returns A constant reference to the MDP associated with this model checker.
                 */
                storm::models::Mdp<Type> const& getModel() const {
                    return AbstractModelChecker<Type>::template getModel<storm::models::Mdp<Type>>();
                }
                
                /*!
                 * Checks the given formula that is a P/R operator without a bound.
                 *
                 * @param formula The formula to check.
                 * @returns The set of states satisfying the formula represented by a bit vector.
                 */
                virtual std::vector<Type>* checkNoBoundOperator(const storm::property::prctl::AbstractNoBoundOperator<Type>& formula) const {
                    // Check if the operator was an non-optimality operator and report an error in that case.
                    if (!formula.isOptimalityOperator()) {
                        LOG4CPLUS_ERROR(logger, "Formula does not specify neither min nor max optimality, which is not meaningful over nondeterministic models.");
                        throw storm::exceptions::InvalidArgumentException() << "Formula does not specify neither min nor max optimality, which is not meaningful over nondeterministic models.";
                    }
                    minimumOperatorStack.push(formula.isMinimumOperator());
                    std::vector<Type>* result = formula.check(*this, false);
                    minimumOperatorStack.pop();
                    return result;
                }
                
                /*!
                 * Checks the given formula that is a bounded-until formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bounds 0 and 1. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bounds 0 and 1.
                 * @returns The probabilities for the given formula to hold on every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact probabilities might not be computed.
                 */
                virtual std::vector<Type>* checkBoundedUntil(const storm::property::prctl::BoundedUntil<Type>& formula, bool qualitative) const {
                    // First, we need to compute the states that satisfy the sub-formulas of the until-formula.
                    storm::storage::BitVector* leftStates = formula.getLeft().check(*this);
                    storm::storage::BitVector* rightStates = formula.getRight().check(*this);
                    std::vector<Type>* result = new std::vector<Type>(this->getModel().getNumberOfStates());
                    
                    // Determine the states that have 0 probability of reaching the target states.
                    storm::storage::BitVector statesWithProbabilityGreater0;
                    if (this->minimumOperatorStack.top()) {
                        statesWithProbabilityGreater0 = storm::utility::graph::performProbGreater0A(this->getModel(), this->getModel().getBackwardTransitions(), *leftStates, *rightStates, true, formula.getBound());
                    } else {
                        statesWithProbabilityGreater0 = storm::utility::graph::performProbGreater0E(this->getModel(), this->getModel().getBackwardTransitions(), *leftStates, *rightStates, true, formula.getBound());
                    }
                    
                    // Check if we already know the result (i.e. probability 0) for all initial states and
                    // don't compute anything in this case.
                    if (this->getInitialStates().isDisjointFrom(statesWithProbabilityGreater0)) {
                        LOG4CPLUS_INFO(logger, "The probabilities for the initial states were determined in a preprocessing step."
                                       << " No exact probabilities were computed.");
                        // Set the values for all maybe-states to 0.5 to indicate that their probability values are not 0 (and
                        // not necessarily 1).
                        storm::utility::vector::setVectorValues(*result, statesWithProbabilityGreater0, Type(0.5));
                    } else {
                        // In this case we have have to compute the probabilities.
                        
                        // We can eliminate the rows and columns from the original transition probability matrix that have probability 0.
                        storm::storage::SparseMatrix<Type> submatrix = this->getModel().getTransitionMatrix().getSubmatrix(statesWithProbabilityGreater0, this->getModel().getNondeterministicChoiceIndices());
                        
                        // Get the "new" nondeterministic choice indices for the submatrix.
                        std::vector<uint_fast64_t> subNondeterministicChoiceIndices = this->computeNondeterministicChoiceIndicesForConstraint(statesWithProbabilityGreater0);
                        
                        // Compute the new set of target states in the reduced system.
                        storm::storage::BitVector rightStatesInReducedSystem = statesWithProbabilityGreater0 % *rightStates;
                        
                        // Make all rows absorbing that satisfy the second sub-formula.
                        submatrix.makeRowsAbsorbing(rightStatesInReducedSystem, subNondeterministicChoiceIndices);
                        
                        // Create the vector with which to multiply.
                        std::vector<Type> subresult(statesWithProbabilityGreater0.getNumberOfSetBits());
                        storm::utility::vector::setVectorValues(subresult, rightStatesInReducedSystem, storm::utility::constGetOne<Type>());
                        
                        if (linearEquationSolver != nullptr) {
                            this->linearEquationSolver->performMatrixVectorMultiplication(this->minimumOperatorStack.top(), submatrix, subresult, subNondeterministicChoiceIndices, nullptr, formula.getBound());
                        } else {
                            throw storm::exceptions::InvalidStateException() << "No valid linear equation solver available.";
                        }
                        
                        // Set the values of the resulting vector accordingly.
                        storm::utility::vector::setVectorValues(*result, statesWithProbabilityGreater0, subresult);
                        storm::utility::vector::setVectorValues(*result, ~statesWithProbabilityGreater0, storm::utility::constGetZero<Type>());
                    }
                    
                    // Delete intermediate results and return result.
                    delete leftStates;
                    delete rightStates;
                    return result;
                }
                
                /*!
                 * Checks the given formula that is a next formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bounds 0 and 1. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bounds 0 and 1.
                 * @returns The probabilities for the given formula to hold on every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact probabilities might not be computed.
                 */
                virtual std::vector<Type>* checkNext(const storm::property::prctl::Next<Type>& formula, bool qualitative) const {
                    // First, we need to compute the states that satisfy the sub-formula of the next-formula.
                    storm::storage::BitVector* nextStates = formula.getChild().check(*this);
                    
                    // Create the vector with which to multiply and initialize it correctly.
                    std::vector<Type>* result = new std::vector<Type>(this->getModel().getNumberOfStates());
                    storm::utility::vector::setVectorValues(*result, *nextStates, storm::utility::constGetOne<Type>());
                    
                    // Delete obsolete sub-result.
                    delete nextStates;
                    
                    if (linearEquationSolver != nullptr) {
                        this->linearEquationSolver->performMatrixVectorMultiplication(this->minimumOperatorStack.top(), this->getModel().getTransitionMatrix(), *result, this->getModel().getNondeterministicChoiceIndices());
                    } else {
                        throw storm::exceptions::InvalidStateException() << "No valid linear equation solver available.";
                    }
                    
                    // Return result.
                    return result;
                }
                
                /*!
                 * Checks the given formula that is a bounded-eventually formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bounds 0 and 1. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bounds 0 and 1.
                 * @returns The probabilities for the given formula to hold on every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact probabilities might not be computed.
                 */
                virtual std::vector<Type>* checkBoundedEventually(const storm::property::prctl::BoundedEventually<Type>& formula, bool qualitative) const {
                    // Create equivalent temporary bounded until formula and check it.
                    storm::property::prctl::BoundedUntil<Type> temporaryBoundedUntilFormula(new storm::property::prctl::Ap<Type>("true"), formula.getChild().clone(), formula.getBound());
                    return this->checkBoundedUntil(temporaryBoundedUntilFormula, qualitative);
                }
                
                /*!
                 * Checks the given formula that is an eventually formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bounds 0 and 1. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bounds 0 and 1.
                 * @returns The probabilities for the given formula to hold on every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact probabilities might not be computed.
                 */
                virtual std::vector<Type>* checkEventually(const storm::property::prctl::Eventually<Type>& formula, bool qualitative) const {
                    // Create equivalent temporary until formula and check it.
                    storm::property::prctl::Until<Type> temporaryUntilFormula(new storm::property::prctl::Ap<Type>("true"), formula.getChild().clone());
                    return this->checkUntil(temporaryUntilFormula, qualitative);
                }
                
                /*!
                 * Checks the given formula that is a globally formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bounds 0 and 1. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bounds 0 and 1.
                 * @returns The probabilities for the given formula to hold on every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact probabilities might not be computed.
                 */
                virtual std::vector<Type>* checkGlobally(const storm::property::prctl::Globally<Type>& formula, bool qualitative) const {
                    // Create "equivalent" temporary eventually formula and check it.
                    storm::property::prctl::Eventually<Type> temporaryEventuallyFormula(new storm::property::prctl::Not<Type>(formula.getChild().clone()));
                    std::vector<Type>* result = this->checkEventually(temporaryEventuallyFormula, qualitative);
                    
                    // Now subtract the resulting vector from the constant one vector to obtain final result.
                    storm::utility::vector::subtractFromConstantOneVector(*result);
                    return result;
                }
                
                /*!
                 * Check the given formula that is an until formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bounds 0 and 1. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bounds 0 and 1.
                 * @param scheduler If <code>qualitative</code> is false and this vector is non-null and has as many elements as
                 * there are states in the MDP, this vector will represent a scheduler for the model that achieves the probability
                 * returned by model checking. To this end, the vector will hold the nondeterministic choice made for each state.
                 * @return The probabilities for the given formula to hold on every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact probabilities might not be computed.
                 */
                virtual std::vector<Type>* checkUntil(const storm::property::prctl::Until<Type>& formula, bool qualitative) const {
                    return this->checkUntil(this->minimumOperatorStack.top(), formula, qualitative, nullptr);
                }
                
                /*!
                 * Check the given formula that is an until formula.
                 *
                 * @param minimize If set, the probability is minimized and maximized otherwise.
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bounds 0 and 1. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bounds 0 and 1.
                 * @param scheduler If <code>qualitative</code> is false and this vector is non-null and has as many elements as
                 * there are states in the MDP, this vector will represent a scheduler for the model that achieves the probability
                 * returned by model checking. To this end, the vector will hold the nondeterministic choice made for each state.
                 * @return The probabilities for the given formula to hold on every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact probabilities might not be computed.
                 */
                virtual std::vector<Type>* checkUntil(bool minimize, const storm::property::prctl::Until<Type>& formula, bool qualitative, std::vector<uint_fast64_t>* scheduler) const {
                    // First, we need to compute the states that satisfy the sub-formulas of the until-formula.
                    storm::storage::BitVector* leftStates = formula.getLeft().check(*this);
                    storm::storage::BitVector* rightStates = formula.getRight().check(*this);
                    
                    // Then, we need to identify the states which have to be taken out of the matrix, i.e.
                    // all states that have probability 0 and 1 of satisfying the until-formula.
                    std::pair<storm::storage::BitVector, storm::storage::BitVector> statesWithProbability01;
                    if (minimize) {
                        statesWithProbability01 = storm::utility::graph::performProb01Min(this->getModel(), *leftStates, *rightStates);
                    } else {
                        statesWithProbability01 = storm::utility::graph::performProb01Max(this->getModel(), *leftStates, *rightStates);
                    }
                    storm::storage::BitVector statesWithProbability0 = std::move(statesWithProbability01.first);
                    storm::storage::BitVector statesWithProbability1 = std::move(statesWithProbability01.second);
                    
                    storm::storage::BitVector maybeStates = ~(statesWithProbability0 | statesWithProbability1);
                    LOG4CPLUS_INFO(logger, "Found " << statesWithProbability0.getNumberOfSetBits() << " 'no' states.");
                    LOG4CPLUS_INFO(logger, "Found " << statesWithProbability1.getNumberOfSetBits() << " 'yes' states.");
                    LOG4CPLUS_INFO(logger, "Found " << maybeStates.getNumberOfSetBits() << " 'maybe' states.");
                    
                    // Create resulting vector.
                    std::vector<Type>* result = new std::vector<Type>(this->getModel().getNumberOfStates());
                    
                    std::vector<uint_fast64_t> guessedScheduler;
                    
                    // Check whether we need to compute exact probabilities for some states.
                    if (this->getInitialStates().isDisjointFrom(maybeStates) || qualitative) {
                        if (qualitative) {
                            LOG4CPLUS_INFO(logger, "The formula was checked qualitatively. No exact probabilities were computed.");
                        } else {
                            LOG4CPLUS_INFO(logger, "The probabilities for the initial states were determined in a preprocessing step."
                                           << " No exact probabilities were computed.");
                        }
                        // Set the values for all maybe-states to 0.5 to indicate that their probability values
                        // are neither 0 nor 1.
                        storm::utility::vector::setVectorValues<Type>(*result, maybeStates, Type(0.5));
                    } else {
                        // In this case we have have to compute the probabilities.
                        
                        // First, we can eliminate the rows and columns from the original transition probability matrix for states
                        // whose probabilities are already known.
                        storm::storage::SparseMatrix<Type> submatrix = this->getModel().getTransitionMatrix().getSubmatrix(maybeStates, this->getModel().getNondeterministicChoiceIndices());
                        
                        // Get the "new" nondeterministic choice indices for the submatrix.
                        std::vector<uint_fast64_t> subNondeterministicChoiceIndices = this->computeNondeterministicChoiceIndicesForConstraint(maybeStates);
                        
                        // Prepare the right-hand side of the equation system. For entry i this corresponds to
                        // the accumulated probability of going from state i to some 'yes' state.
                        std::vector<Type> b = this->getModel().getTransitionMatrix().getConstrainedRowSumVector(maybeStates, this->getModel().getNondeterministicChoiceIndices(), statesWithProbability1, submatrix.getRowCount());
                        
                        // Create vector for results for maybe states.
                        std::vector<Type> x = this->getInitialValueIterationValues(minimize, submatrix, subNondeterministicChoiceIndices, b, statesWithProbability1, maybeStates, &guessedScheduler);
                                                
                        // Solve the corresponding system of equations.
                        if (linearEquationSolver != nullptr) {
                            this->linearEquationSolver->solveEquationSystem(minimize, submatrix, x, b, subNondeterministicChoiceIndices);
                        } else {
                            throw storm::exceptions::InvalidStateException() << "No valid linear equation solver available.";
                        }
                        
                        // Set values of resulting vector according to result.
                        storm::utility::vector::setVectorValues<Type>(*result, maybeStates, x);
                    }
                    
                    // Set values of resulting vector that are known exactly.
                    storm::utility::vector::setVectorValues<Type>(*result, statesWithProbability0, storm::utility::constGetZero<Type>());
                    storm::utility::vector::setVectorValues<Type>(*result, statesWithProbability1, storm::utility::constGetOne<Type>());
                    
                    // Output a scheduler for debug purposes.
                    std::vector<uint_fast64_t> myScheduler(this->getModel().getNumberOfStates());
                    this->computeTakenChoices(this->minimumOperatorStack.top(), false, *result, myScheduler, this->getModel().getNondeterministicChoiceIndices());
                    
                    std::vector<uint_fast64_t> stateColoring(this->getModel().getNumberOfStates());
                    for (auto target : statesWithProbability1) {
                        stateColoring[target] = 1;
                    }
                    
                    std::vector<std::string> colors(2);
                    colors[0] = "white";
                    colors[1] = "blue";
                    
                    std::ofstream outFile;
                    outFile.open("real.dot");
                    storm::storage::BitVector filterStates(this->getModel().getNumberOfStates(), true);
                    this->getModel().writeDotToStream(outFile, true, &filterStates, result, nullptr, &stateColoring, &colors, &myScheduler);
                    outFile.close();
                    
                    std::cout << "=========== Scheduler Comparison ===========" << std::endl;
                    uint_fast64_t index = 0;
                    for (auto state : maybeStates) {
                        if (myScheduler[state] != guessedScheduler[index]) {
                            std::cout << state << " right: " << myScheduler[state] << ", wrong: " << guessedScheduler[index] << std::endl;
                            std::cout << "Correct: " << std::endl;
                            typename storm::storage::SparseMatrix<Type>::Rows correctRow = this->getModel().getTransitionMatrix().getRow(this->getModel().getNondeterministicChoiceIndices()[state] + myScheduler[state]);
                            for (auto& transition : correctRow) {
                                std::cout << "target " << transition.column() << " with prob " << transition.value() << std::endl;
                            }
                            std::cout << "Incorrect: " << std::endl;
                            typename storm::storage::SparseMatrix<Type>::Rows incorrectRow = this->getModel().getTransitionMatrix().getRow(this->getModel().getNondeterministicChoiceIndices()[state] + myScheduler[state]);
                            for (auto& transition : incorrectRow) {
                                std::cout << "target " << transition.column() << " with prob " << transition.value() << std::endl;
                            }
                        }
                        ++index;
                    }
                    
                    // If we were required to generate a scheduler, do so now.
                    if (scheduler != nullptr) {
                        this->computeTakenChoices(this->minimumOperatorStack.top(), false, *result, *scheduler, this->getModel().getNondeterministicChoiceIndices());
                    }
                    
                    // Delete sub-results that are obsolete now.
                    delete leftStates;
                    delete rightStates;
                    return result;
                }
                
                /*!
                 * Checks the given formula that is an instantaneous reward formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bound 0. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bound 0.
                 * @returns The reward values for the given formula for every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact values might not be computed.
                 */
                virtual std::vector<Type>* checkInstantaneousReward(const storm::property::prctl::InstantaneousReward<Type>& formula, bool qualitative) const {
                    // Only compute the result if the model has a state-based reward model.
                    if (!this->getModel().hasStateRewards()) {
                        LOG4CPLUS_ERROR(logger, "Missing (state-based) reward model for formula.");
                        throw storm::exceptions::InvalidPropertyException() << "Missing (state-based) reward model for formula.";
                    }
                    
                    // Initialize result to state rewards of the model.
                    std::vector<Type>* result = new std::vector<Type>(this->getModel().getStateRewardVector());
                    
                    if (linearEquationSolver != nullptr) {
                        this->linearEquationSolver->performMatrixVectorMultiplication(this->minimumOperatorStack.top(), this->getModel().getTransitionMatrix(), *result, this->getModel().getNondeterministicChoiceIndices(), nullptr, formula.getBound());
                    } else {
                        throw storm::exceptions::InvalidStateException() << "No valid linear equation solver available.";
                    }
                    
                    // Return result.
                    return result;
                }
                
                /*!
                 * Check the given formula that is a cumulative reward formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bound 0. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bound 0.
                 * @returns The reward values for the given formula for every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact values might not be computed.
                 */
                virtual std::vector<Type>* checkCumulativeReward(const storm::property::prctl::CumulativeReward<Type>& formula, bool qualitative) const {
                    // Only compute the result if the model has at least one reward model.
                    if (!this->getModel().hasStateRewards() && !this->getModel().hasTransitionRewards()) {
                        LOG4CPLUS_ERROR(logger, "Missing reward model for formula.");
                        throw storm::exceptions::InvalidPropertyException() << "Missing reward model for formula.";
                    }
                    
                    // Compute the reward vector to add in each step based on the available reward models.
                    std::vector<Type> totalRewardVector;
                    if (this->getModel().hasTransitionRewards()) {
                        totalRewardVector = this->getModel().getTransitionMatrix().getPointwiseProductRowSumVector(this->getModel().getTransitionRewardMatrix());
                        if (this->getModel().hasStateRewards()) {
                            storm::utility::vector::addVectorsInPlace(totalRewardVector, this->getModel().getStateRewardVector());
                        }
                    } else {
                        totalRewardVector = std::vector<Type>(this->getModel().getStateRewardVector());
                    }
                    
                    // Initialize result to either the state rewards of the model or the null vector.
                    std::vector<Type>* result = nullptr;
                    if (this->getModel().hasStateRewards()) {
                        result = new std::vector<Type>(this->getModel().getStateRewardVector());
                    } else {
                        result = new std::vector<Type>(this->getModel().getNumberOfStates());
                    }
                    
                    if (linearEquationSolver != nullptr) {
                        this->linearEquationSolver->performMatrixVectorMultiplication(this->minimumOperatorStack.top(), this->getModel().getTransitionMatrix(), *result, this->getModel().getNondeterministicChoiceIndices(), &totalRewardVector, formula.getBound());
                    } else {
                        throw storm::exceptions::InvalidStateException() << "No valid linear equation solver available.";
                    }
                    
                    // Delete temporary variables and return result.
                    return result;
                }
                
                /*!
                 * Checks the given formula that is a reachability reward formula.
                 *
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bound 0. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bound 0.
                 * @return The reward values for the given formula for every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact values might not be computed.
                 */
                virtual std::vector<Type>* checkReachabilityReward(const storm::property::prctl::ReachabilityReward<Type>& formula, bool qualitative) const {
                    return this->checkReachabilityReward(this->minimumOperatorStack.top(), formula, qualitative, nullptr);
                }
                
                /*!
                 * Checks the given formula that is a reachability reward formula.
                 *
                 * @param minimize If set, the reward is to be minimized and maximized otherwise.
                 * @param formula The formula to check.
                 * @param qualitative A flag indicating whether the formula only needs to be evaluated qualitatively, i.e. if the
                 * results are only compared against the bound 0. If set to true, this will most likely results that are only
                 * qualitatively correct, i.e. do not represent the correct value, but only the correct relation with respect to the
                 * bound 0.
                 * @param scheduler If <code>qualitative</code> is false and this vector is non-null and has as many elements as
                 * there are states in the MDP, this vector will represent a scheduler for the model that achieves the probability
                 * returned by model checking. To this end, the vector will hold the nondeterministic choice made for each state.
                 * @return The reward values for the given formula for every state of the model associated with this model
                 * checker. If the qualitative flag is set, exact values might not be computed.
                 */
                virtual std::vector<Type>* checkReachabilityReward(bool minimize, const storm::property::prctl::ReachabilityReward<Type>& formula, bool qualitative, std::vector<uint_fast64_t>* scheduler) const {
                    // Only compute the result if the model has at least one reward model.
                    if (!this->getModel().hasStateRewards() && !this->getModel().hasTransitionRewards()) {
                        LOG4CPLUS_ERROR(logger, "Missing reward model for formula. Skipping formula");
                        throw storm::exceptions::InvalidPropertyException() << "Missing reward model for formula.";
                    }
                    
                    // Determine the states for which the target predicate holds.
                    storm::storage::BitVector* targetStates = formula.getChild().check(*this);
                    
                    // Determine which states have a reward of infinity by definition.
                    storm::storage::BitVector infinityStates;
                    storm::storage::BitVector trueStates(this->getModel().getNumberOfStates(), true);
                    if (minimize) {
                        infinityStates = std::move(storm::utility::graph::performProb1A(this->getModel(), this->getModel().getBackwardTransitions(), trueStates, *targetStates));
                    } else {
                        infinityStates = std::move(storm::utility::graph::performProb1E(this->getModel(), this->getModel().getBackwardTransitions(), trueStates, *targetStates));
                    }
                    infinityStates.complement();
                    storm::storage::BitVector maybeStates = ~(*targetStates) & ~infinityStates;
                    LOG4CPLUS_INFO(logger, "Found " << infinityStates.getNumberOfSetBits() << " 'infinity' states.");
                    LOG4CPLUS_INFO(logger, "Found " << targetStates->getNumberOfSetBits() << " 'target' states.");
                    LOG4CPLUS_INFO(logger, "Found " << maybeStates.getNumberOfSetBits() << " 'maybe' states.");
                    
                    // Create resulting vector.
                    std::vector<Type>* result = new std::vector<Type>(this->getModel().getNumberOfStates());
                    
                    // Check whether we need to compute exact rewards for some states.
                    if (this->getInitialStates().isDisjointFrom(maybeStates)) {
                        LOG4CPLUS_INFO(logger, "The rewards for the initial states were determined in a preprocessing step."
                                       << " No exact rewards were computed.");
                        // Set the values for all maybe-states to 1 to indicate that their reward values
                        // are neither 0 nor infinity.
                        storm::utility::vector::setVectorValues<Type>(*result, maybeStates, storm::utility::constGetOne<Type>());
                    } else {
                        // In this case we have to compute the reward values for the remaining states.
                        
                        // We can eliminate the rows and columns from the original transition probability matrix for states
                        // whose reward values are already known.
                        storm::storage::SparseMatrix<Type> submatrix = this->getModel().getTransitionMatrix().getSubmatrix(maybeStates, this->getModel().getNondeterministicChoiceIndices());
                        
                        // Get the "new" nondeterministic choice indices for the submatrix.
                        std::vector<uint_fast64_t> subNondeterministicChoiceIndices = this->computeNondeterministicChoiceIndicesForConstraint(maybeStates);
                        
                        // Prepare the right-hand side of the equation system. For entry i this corresponds to
                        // the accumulated probability of going from state i to some 'yes' state.
                        std::vector<Type> b(submatrix.getRowCount());
                        
                        if (this->getModel().hasTransitionRewards()) {
                            // If a transition-based reward model is available, we initialize the right-hand
                            // side to the vector resulting from summing the rows of the pointwise product
                            // of the transition probability matrix and the transition reward matrix.
                            std::vector<Type> pointwiseProductRowSumVector = this->getModel().getTransitionMatrix().getPointwiseProductRowSumVector(this->getModel().getTransitionRewardMatrix());
                            storm::utility::vector::selectVectorValues(b, maybeStates, this->getModel().getNondeterministicChoiceIndices(), pointwiseProductRowSumVector);
                            
                            if (this->getModel().hasStateRewards()) {
                                // If a state-based reward model is also available, we need to add this vector
                                // as well. As the state reward vector contains entries not just for the states
                                // that we still consider (i.e. maybeStates), we need to extract these values
                                // first.
                                std::vector<Type> subStateRewards(b.size());
                                storm::utility::vector::selectVectorValuesRepeatedly(subStateRewards, maybeStates, this->getModel().getNondeterministicChoiceIndices(), this->getModel().getStateRewardVector());
                                storm::utility::vector::addVectorsInPlace(b, subStateRewards);
                            }
                        } else {
                            // If only a state-based reward model is  available, we take this vector as the
                            // right-hand side. As the state reward vector contains entries not just for the
                            // states that we still consider (i.e. maybeStates), we need to extract these values
                            // first.
                            storm::utility::vector::selectVectorValuesRepeatedly(b, maybeStates, this->getModel().getNondeterministicChoiceIndices(), this->getModel().getStateRewardVector());
                        }
                        
                        // Create vector for results for maybe states.
                        std::vector<Type> x = this->getInitialValueIterationValues(minimize, submatrix, subNondeterministicChoiceIndices, b, *targetStates, maybeStates);
                        
                        // Solve the corresponding system of equations.
                        if (linearEquationSolver != nullptr) {
                            this->linearEquationSolver->solveEquationSystem(minimize, submatrix, x, b, subNondeterministicChoiceIndices);
                        } else {
                            throw storm::exceptions::InvalidStateException() << "No valid linear equation solver available.";
                        }
                        
                        // Set values of resulting vector according to result.
                        storm::utility::vector::setVectorValues<Type>(*result, maybeStates, x);
                    }
                    
                    // Set values of resulting vector that are known exactly.
                    storm::utility::vector::setVectorValues(*result, *targetStates, storm::utility::constGetZero<Type>());
                    storm::utility::vector::setVectorValues(*result, infinityStates, storm::utility::constGetInfinity<Type>());
                    
//                    std::vector<uint_fast64_t> myScheduler(this->getModel().getNumberOfStates());
//                    this->computeTakenChoices(this->minimumOperatorStack.top(), true, *result, myScheduler, this->getModel().getNondeterministicChoiceIndices());
//                    
//                    std::vector<uint_fast64_t> stateColoring(this->getModel().getNumberOfStates());
//                    for (auto target : *targetStates) {
//                        stateColoring[target] = 1;
//                    }
//                    
//                    std::vector<std::string> colors(2);
//                    colors[0] = "white";
//                    colors[1] = "blue";
//                    
//                    this->getModel().writeDotToStream(std::cout, true, storm::storage::BitVector(this->getModel().getNumberOfStates(), true), result, nullptr, &stateColoring, &colors, &myScheduler);
                    
                    // If we were required to generate a scheduler, do so now.
                    if (scheduler != nullptr) {
                        this->computeTakenChoices(this->minimumOperatorStack.top(), true, *result, *scheduler, this->getModel().getNondeterministicChoiceIndices());
                    }
                    
                    // Delete temporary storages and return result.
                    delete targetStates;
                    return result;
                }
                
            protected:
                /*!
                 * Computes the vector of choices that need to be made to minimize/maximize the model checking result for each state.
                 *
                 * @param minimize If set, all choices are resolved such that the solution value becomes minimal and maximal otherwise.
                 * @param nondeterministicResult The model checking result for nondeterministic choices of all states.
                 * @param takenChoices The output vector that is to store the taken choices.
                 * @param nondeterministicChoiceIndices The assignment of states to their nondeterministic choices in the matrix.
                 */
                void computeTakenChoices(bool minimize, bool addRewards, std::vector<Type> const& result, std::vector<uint_fast64_t>& takenChoices, std::vector<uint_fast64_t> const& nondeterministicChoiceIndices) const {
                    std::vector<Type> temporaryResult(nondeterministicChoiceIndices.size() - 1);
                    std::vector<Type> nondeterministicResult(result);
                    storm::solver::GmmxxLinearEquationSolver<Type> solver;
                    solver.performMatrixVectorMultiplication(this->getModel().getTransitionMatrix(), nondeterministicResult, nullptr, 1);
                    if (addRewards) {
                        std::vector<Type> totalRewardVector;
                        if (this->getModel().hasTransitionRewards()) {
                            std::vector<Type> totalRewardVector = this->getModel().getTransitionMatrix().getPointwiseProductRowSumVector(this->getModel().getTransitionRewardMatrix());
                            if (this->getModel().hasStateRewards()) {
                                std::vector<Type> stateRewards(totalRewardVector.size());
                                storm::utility::vector::selectVectorValuesRepeatedly(stateRewards, storm::storage::BitVector(this->getModel().getStateRewardVector().size(), true), this->getModel().getNondeterministicChoiceIndices(), this->getModel().getStateRewardVector());
                                storm::utility::vector::addVectorsInPlace(totalRewardVector, stateRewards);
                            }
                        } else {
                            totalRewardVector.resize(nondeterministicResult.size());
                            storm::utility::vector::selectVectorValuesRepeatedly(totalRewardVector, storm::storage::BitVector(this->getModel().getStateRewardVector().size(), true), this->getModel().getNondeterministicChoiceIndices(), this->getModel().getStateRewardVector());
                        }
                        storm::utility::vector::addVectorsInPlace(nondeterministicResult, totalRewardVector);
                    }
                    if (minimize) {
                        storm::utility::vector::reduceVectorMin(nondeterministicResult, temporaryResult, nondeterministicChoiceIndices, &takenChoices);
                    } else {
                        storm::utility::vector::reduceVectorMax(nondeterministicResult, temporaryResult, nondeterministicChoiceIndices, &takenChoices);
                    }
                }
                
                /*!
                 * A stack used for storing whether we are currently computing min or max probabilities or rewards, respectively.
                 * The topmost element is true if and only if we are currently computing minimum probabilities or rewards.
                 */
                mutable std::stack<bool> minimumOperatorStack;
                
            private:
                /*!
                 * Computes the nondeterministic choice indices vector resulting from reducing the full system to the states given
                 * by the parameter constraint.
                 *
                 * @param constraint A bit vector specifying which states are kept.
                 * @returns A vector of the nondeterministic choice indices of the subsystem induced by the given constraint.
                 */
                std::vector<uint_fast64_t> computeNondeterministicChoiceIndicesForConstraint(storm::storage::BitVector const& constraint) const {
                    // First, get a reference to the full nondeterministic choice indices.
                    std::vector<uint_fast64_t> const& nondeterministicChoiceIndices = this->getModel().getNondeterministicChoiceIndices();
                    
                    // Reserve the known amount of slots for the resulting vector.
                    std::vector<uint_fast64_t> subNondeterministicChoiceIndices(constraint.getNumberOfSetBits() + 1);
                    uint_fast64_t currentRowCount = 0;
                    uint_fast64_t currentIndexCount = 1;
                    
                    // Set the first element as this will clearly begin at offset 0.
                    subNondeterministicChoiceIndices[0] = 0;
                    
                    // Loop over all states that need to be kept and copy the relative indices of the nondeterministic choices over
                    // to the resulting vector.
                    for (auto index : constraint) {
                        subNondeterministicChoiceIndices[currentIndexCount] = currentRowCount + nondeterministicChoiceIndices[index + 1] - nondeterministicChoiceIndices[index];
                        currentRowCount += nondeterministicChoiceIndices[index + 1] - nondeterministicChoiceIndices[index];
                        ++currentIndexCount;
                    }
                    
                    // Put a sentinel element at the end.
                    subNondeterministicChoiceIndices[constraint.getNumberOfSetBits()] = currentRowCount;
                    
                    return subNondeterministicChoiceIndices;
                }
                
                /*!
                 * Retrieves the values to be used as the initial values for the value iteration techniques.
                 *
                 * @param submatrix The matrix that will be used for value iteration later.
                 */
                std::vector<Type> getInitialValueIterationValues(bool minimize, storm::storage::SparseMatrix<Type> const& submatrix,
                                                                 std::vector<uint_fast64_t> const& subNondeterministicChoiceIndices,
                                                                 std::vector<Type> const& rightHandSide,
                                                                 storm::storage::BitVector const& targetStates,
                                                                 storm::storage::BitVector const& maybeStates,
                                                                 std::vector<uint_fast64_t>* guessedScheduler = nullptr) const {
                    storm::settings::Settings* s = storm::settings::instance();
                    double precision = s->get<double>("precision");
                    if (s->get<bool>("use-heuristic-presolve")) {
                        std::pair<std::vector<Type>, std::vector<uint_fast64_t>> distancesAndPredecessorsPair = storm::utility::graph::performDijkstra(this->getModel(),
                                                                                                                                                       this->getModel().template getBackwardTransitions<Type>([](Type const& value) -> Type { return value; }),
                                                                                                                                                       minimize ? ~(maybeStates | targetStates) : targetStates, &maybeStates);
                        
                        std::pair<std::vector<Type>, std::vector<uint_fast64_t>> distancesAndPredecessorsPair2 = storm::utility::graph::performDijkstra(this->getModel(),
                                                                                                                                                       this->getModel().template getBackwardTransitions<Type>([](Type const& value) -> Type { return value; }),
                                                                                                                                                       minimize ? targetStates : ~(maybeStates | targetStates), &maybeStates);
                        
                        std::vector<uint_fast64_t> scheduler = this->convertShortestPathsToScheduler(false, maybeStates, distancesAndPredecessorsPair.first, distancesAndPredecessorsPair.second);
                                                
                        std::vector<uint_fast64_t> stateColoring(this->getModel().getNumberOfStates());
                        for (auto target : targetStates) {
                            stateColoring[target] = 1;
                        }
                        
                        std::vector<std::string> colors(2);
                        colors[0] = "white";
                        colors[1] = "blue";
                        
                        std::ofstream outFile;
                        outFile.open("guessed.dot");
                        storm::storage::BitVector filterStates(this->getModel().getNumberOfStates(), true);
                        this->getModel().writeDotToStream(outFile, true, &filterStates, &distancesAndPredecessorsPair.first, &distancesAndPredecessorsPair2.first, &stateColoring, &colors);
                        outFile.close();
                        
                        std::vector<Type> result(scheduler.size(), Type(0.5));
                        std::vector<Type> b(scheduler.size());
                        storm::utility::vector::selectVectorValues(b, scheduler, subNondeterministicChoiceIndices, rightHandSide);
                        storm::storage::SparseMatrix<Type> A(submatrix.getSubmatrix(scheduler, subNondeterministicChoiceIndices));
                        A.convertToEquationSystem();
                        storm::solver::GmmxxLinearEquationSolver<Type> solver;
                        solver.solveEquationSystem(A, result, b);
                        
                        // As there are sometimes some very small values in the vector due to numerical solving, we set
                        // them to zero, because they otherwise require a certain number of value iterations.
                        for (auto& value : result) {
                            if (value < precision) {
                                value = 0;
                            }
                        }
                        
                        if (guessedScheduler != nullptr) {
                            *guessedScheduler = std::move(scheduler);
                        }
                        
                        return result;
                    } else {
                        return std::vector<Type>(submatrix.getColumnCount());
                    }
                }
                
                std::vector<uint_fast64_t> convertShortestPathsToScheduler(bool minimize, storm::storage::BitVector const& maybeStates, std::vector<Type> const& distances, std::vector<uint_fast64_t> const& shortestPathSuccessors) const {
                    std::vector<uint_fast64_t> scheduler(maybeStates.getNumberOfSetBits());

                    Type extremalProbability = minimize ? 1 : 0;
                    Type currentProbability = 0;
                    uint_fast64_t currentStateIndex = 0;
                    for (auto state : maybeStates) {
                        extremalProbability = minimize ? 1 : 0;
                        
                        for (uint_fast64_t row = 0, rowEnd = this->getModel().getNondeterministicChoiceIndices()[state + 1] - this->getModel().getNondeterministicChoiceIndices()[state]; row < rowEnd; ++row) {
                            typename storm::storage::SparseMatrix<Type>::Rows currentRow = this->getModel().getTransitionMatrix().getRow(this->getModel().getNondeterministicChoiceIndices()[state] + row);
                            currentProbability = 0;
                            
                            for (auto& transition : currentRow) {
                                currentProbability += transition.value() * (1 / std::exp(distances[transition.column()]));
                            }
                            
                            if (minimize && currentProbability < extremalProbability) {
                                extremalProbability = currentProbability;
                                scheduler[currentStateIndex] = row;
                            } else if (!minimize && currentProbability > extremalProbability) {
                                extremalProbability = currentProbability;
                                scheduler[currentStateIndex] = row;
                            }
                        }
                        
                        ++currentStateIndex;
                    }
                    
                    return scheduler;
                }
                
                // An object that is used for solving linear equations and performing matrix-vector multiplication.
                std::unique_ptr<storm::solver::AbstractNondeterministicLinearEquationSolver<Type>> linearEquationSolver;
                
            };
            
        } // namespace prctl
    } // namespace modelchecker
} // namespace storm

#endif /* STORM_MODELCHECKER_PRCTL_SPARSEMDPPRCTLMODELCHECKER_H_ */
