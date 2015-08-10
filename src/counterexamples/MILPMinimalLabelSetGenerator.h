#ifndef STORM_COUNTEREXAMPLES_MILPMINIMALLABELSETGENERATOR_MDP_H_
#define STORM_COUNTEREXAMPLES_MILPMINIMALLABELSETGENERATOR_MDP_H_

#include <chrono>

#include "src/models/sparse/Mdp.h"
#include "src/logic/Formulas.h"
#include "src/storage/prism/Program.h"
#include "src/modelchecker/prctl/SparseMdpPrctlModelChecker.h"
#include "src/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "src/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "src/exceptions/NotImplementedException.h"
#include "src/exceptions/InvalidPropertyException.h"
#include "src/exceptions/InvalidArgumentException.h"
#include "src/exceptions/InvalidStateException.h"

#include "src/utility/graph.h"
#include "src/utility/counterexamples.h"
#include "src/utility/solver.h"

namespace storm {
    namespace counterexamples {
        
        /*!
         * This class provides functionality to generate a minimal counterexample to a probabilistic reachability
         * property in terms of used labels.
         */
        template <class T>
        class MILPMinimalLabelSetGenerator {
        private:
            /*!
             * A helper class that provides the functionality to compute a hash value for pairs of state indices.
             */
            class PairHash {
            public:
                std::size_t operator()(std::pair<uint_fast64_t, uint_fast64_t> const& pair) const {
                    size_t seed = 0;
                    boost::hash_combine(seed, pair.first);
                    boost::hash_combine(seed, pair.second);
                    return seed;
                }
            };
            
            /*!
             * A helper struct storing which states are relevant or problematic.
             */
            struct StateInformation {
                storm::storage::BitVector relevantStates;
                storm::storage::BitVector problematicStates;
            };
            
            /*!
             * A helper struct capturing information about relevant and problematic choices of states and which labels
             * are relevant.
             */
            struct ChoiceInformation {
                std::unordered_map<uint_fast64_t, std::list<uint_fast64_t>> relevantChoicesForRelevantStates;
                std::unordered_map<uint_fast64_t, std::list<uint_fast64_t>> problematicChoicesForProblematicStates;
                boost::container::flat_set<uint_fast64_t> allRelevantLabels;
                boost::container::flat_set<uint_fast64_t> knownLabels;
            };

            /*!
             * A helper struct capturing information about the variables of the MILP formulation.
             */
            struct VariableInformation {
                std::unordered_map<uint_fast64_t, storm::expressions::Variable> labelToVariableMap;
                std::unordered_map<uint_fast64_t, std::list<storm::expressions::Variable>> stateToChoiceVariablesMap;
                std::unordered_map<uint_fast64_t, storm::expressions::Variable> initialStateToChoiceVariableMap;
                std::unordered_map<uint_fast64_t, storm::expressions::Variable> stateToProbabilityVariableMap;
                storm::expressions::Variable virtualInitialStateVariable;
                std::unordered_map<uint_fast64_t, storm::expressions::Variable> problematicStateToVariableMap;
                std::unordered_map<std::pair<uint_fast64_t, uint_fast64_t>, storm::expressions::Variable, PairHash> problematicTransitionToVariableMap;
                uint_fast64_t numberOfVariables;

				VariableInformation() : numberOfVariables(0) {}
            };

            /*!
             * Determines the relevant and the problematic states of the given MDP with respect to the given phi and psi
             * state sets. The relevant states are those for which there exists at least one scheduler that attains a
             * non-zero probability of satisfying phi until psi. Problematic states are relevant states that have at
             * least one scheduler such that the probability of satisfying phi until psi is zero.
             *
             * @param labeledMdp The MDP whose states to search.
             * @param phiStates A bit vector characterizing all states satisfying phi.
             * @param psiStates A bit vector characterizing all states satisfying psi.
             * @return A structure that stores the relevant and problematic states.
             */
            static struct StateInformation determineRelevantAndProblematicStates(storm::models::sparse::Mdp<T> const& labeledMdp, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates) {
                StateInformation result;
                result.relevantStates = storm::utility::graph::performProbGreater0E(labeledMdp.getTransitionMatrix(), labeledMdp.getNondeterministicChoiceIndices(), labeledMdp.getBackwardTransitions(), phiStates, psiStates);
                result.relevantStates &= ~psiStates;
                result.problematicStates = storm::utility::graph::performProb0E(labeledMdp.getTransitionMatrix(), labeledMdp.getNondeterministicChoiceIndices(), labeledMdp.getBackwardTransitions(), phiStates, psiStates);
                result.problematicStates &= result.relevantStates;
                LOG4CPLUS_DEBUG(logger, "Found " << phiStates.getNumberOfSetBits() << " filter states.");
                LOG4CPLUS_DEBUG(logger, "Found " << psiStates.getNumberOfSetBits() << " target states.");
                LOG4CPLUS_DEBUG(logger, "Found " << result.relevantStates.getNumberOfSetBits() << " relevant states.");
                LOG4CPLUS_DEBUG(logger, "Found " << result.problematicStates.getNumberOfSetBits() << " problematic states.");
                return result;
            }
            
            /*!
             * Determines the relevant and problematic choices of the given MDP with respect to the given parameters.
             *
             * @param labeledMdp The MDP whose choices to search.
             * @param stateInformation The relevant and problematic states of the model.
             * @param psiStates A bit vector characterizing the psi states in the model.
             * @return A structure that stores the relevant and problematic choices in the model as well as the set
             * of relevant labels.
             */
            static struct ChoiceInformation determineRelevantAndProblematicChoices(storm::models::sparse::Mdp<T> const& labeledMdp, StateInformation const& stateInformation, storm::storage::BitVector const& psiStates) {
                // Create result and shortcuts to needed data for convenience.
                ChoiceInformation result;
                storm::storage::SparseMatrix<T> const& transitionMatrix = labeledMdp.getTransitionMatrix();
                std::vector<uint_fast64_t> const& nondeterministicChoiceIndices = labeledMdp.getNondeterministicChoiceIndices();
                std::vector<boost::container::flat_set<uint_fast64_t>> const& choiceLabeling = labeledMdp.getChoiceLabeling();
                
                // Now traverse all choices of all relevant states and check whether there is a relevant target state.
                // If so, the associated labels become relevant. Also, if a choice of relevant state has at least one
                // relevant successor, the choice is considered to be relevant.
                for (auto state : stateInformation.relevantStates) {
                    result.relevantChoicesForRelevantStates.emplace(state, std::list<uint_fast64_t>());
                    if (stateInformation.problematicStates.get(state)) {
                        result.problematicChoicesForProblematicStates.emplace(state, std::list<uint_fast64_t>());
                    }
                    for (uint_fast64_t row = nondeterministicChoiceIndices[state]; row < nondeterministicChoiceIndices[state + 1]; ++row) {
                        bool currentChoiceRelevant = false;
                        bool allSuccessorsProblematic = true;
                        for (auto const& successorEntry : transitionMatrix.getRow(row)) {
                            // If there is a relevant successor, we need to add the labels of the current choice.
                            if (stateInformation.relevantStates.get(successorEntry.getColumn()) || psiStates.get(successorEntry.getColumn())) {
                                for (auto const& label : choiceLabeling[row]) {
                                    result.allRelevantLabels.insert(label);
                                }
                                if (!currentChoiceRelevant) {
                                    currentChoiceRelevant = true;
                                    result.relevantChoicesForRelevantStates[state].push_back(row);
                                }
                            }
                            if (!stateInformation.problematicStates.get(successorEntry.getColumn())) {
                                allSuccessorsProblematic = false;
                            }
                        }
                        
                        // If all successors of a problematic state are problematic themselves, we record this choice
                        // as being problematic.
                        if (stateInformation.problematicStates.get(state) && allSuccessorsProblematic) {
                            result.problematicChoicesForProblematicStates[state].push_back(row);
                        }
                    }
                }

                // Finally, determine the set of labels that are known to be taken.
                result.knownLabels = storm::utility::counterexamples::getGuaranteedLabelSet(labeledMdp, psiStates, result.allRelevantLabels);
                LOG4CPLUS_DEBUG(logger, "Found " << result.allRelevantLabels.size() << " relevant labels and " << result.knownLabels.size() << " known labels.");

                return result;
            }

            /*!
             * Creates the variables for the labels of the model.
             *
             * @param solver The MILP solver.
             * @param relevantLabels The set of relevant labels of the model.
             * @return A mapping from labels to variable indices.
             */
            static std::pair<std::unordered_map<uint_fast64_t, storm::expressions::Variable>, uint_fast64_t> createLabelVariables(storm::solver::LpSolver& solver, boost::container::flat_set<uint_fast64_t> const& relevantLabels) {
                std::stringstream variableNameBuffer;
                std::unordered_map<uint_fast64_t, storm::expressions::Variable> resultingMap;
                for (auto const& label : relevantLabels) {
                    variableNameBuffer.str("");
                    variableNameBuffer.clear();
                    variableNameBuffer << "label" << label;
                    resultingMap[label] = solver.addBinaryVariable(variableNameBuffer.str(), 1);
                }
                return std::make_pair(resultingMap, relevantLabels.size());
            }
            
            /*!
             * Creates the variables for the relevant choices in the model.
             *
             * @param solver The MILP solver.
             * @param stateInformation The information about the states of the model.
             * @param choiceInformation The information about the choices of the model.
             * @return A mapping from states to a list of choice variable indices.
             */
            static std::pair<std::unordered_map<uint_fast64_t, std::list<storm::expressions::Variable>>, uint_fast64_t> createSchedulerVariables(storm::solver::LpSolver& solver, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation) {
                std::stringstream variableNameBuffer;
                uint_fast64_t numberOfVariablesCreated = 0;
                std::unordered_map<uint_fast64_t, std::list<storm::expressions::Variable>> resultingMap;
                
                for (auto state : stateInformation.relevantStates) {
                    resultingMap.emplace(state, std::list<storm::expressions::Variable>());
                    std::list<uint_fast64_t> const& relevantChoicesForState = choiceInformation.relevantChoicesForRelevantStates.at(state);
                    for (uint_fast64_t row : relevantChoicesForState) {
                        variableNameBuffer.str("");
                        variableNameBuffer.clear();
                        variableNameBuffer << "choice" << row << "in" << state;
                        resultingMap[state].push_back(solver.addBinaryVariable(variableNameBuffer.str()));
                        ++numberOfVariablesCreated;
                    }
                }
                return std::make_pair(resultingMap, numberOfVariablesCreated);
            }
            
            /*!
             * Creates the variables needed for encoding the nondeterministic selection of one of the initial states
             * in the model.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param stateInformation The information about the states of the model.
             * @return A mapping from initial states to choice variable indices.
             */
            static std::pair<std::unordered_map<uint_fast64_t, storm::expressions::Variable>, uint_fast64_t> createInitialChoiceVariables(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, StateInformation const& stateInformation) {
                std::stringstream variableNameBuffer;
                uint_fast64_t numberOfVariablesCreated = 0;
                std::unordered_map<uint_fast64_t, storm::expressions::Variable> resultingMap;
                
                for (auto initialState : labeledMdp.getStates("init")) {
                    // Only consider this initial state if it is relevant.
                    if (stateInformation.relevantStates.get(initialState)) {
                        variableNameBuffer.str("");
                        variableNameBuffer.clear();
                        variableNameBuffer << "init" << initialState;
                        resultingMap[initialState] = solver.addBinaryVariable(variableNameBuffer.str());
                        ++numberOfVariablesCreated;
                    }
                }
                return std::make_pair(resultingMap, numberOfVariablesCreated);
            }
            
            /*!
             * Creates the variables for the probabilities in the model.
             *
             * @param solver The MILP solver.
             * @param stateInformation The information about the states in the model.
             * @return A mapping from states to the index of the corresponding probability variables.
             */
            static std::pair<std::unordered_map<uint_fast64_t, storm::expressions::Variable>, uint_fast64_t> createProbabilityVariables(storm::solver::LpSolver& solver, StateInformation const& stateInformation) {
                std::stringstream variableNameBuffer;
                uint_fast64_t numberOfVariablesCreated = 0;
                std::unordered_map<uint_fast64_t, storm::expressions::Variable> resultingMap;
                
                for (auto state : stateInformation.relevantStates) {
                    variableNameBuffer.str("");
                    variableNameBuffer.clear();
                    variableNameBuffer << "p" << state;
                    resultingMap[state] = solver.addBoundedContinuousVariable(variableNameBuffer.str(), 0, 1);
                    ++numberOfVariablesCreated;
                }
                return std::make_pair(resultingMap, numberOfVariablesCreated);
            }
            
            /*!
             * Creates the variable for the probability of the virtual initial state.
             *
             * @param solver The MILP solver.
             * @param maximizeProbability If set to true, the objective function is constructed in a way that a
             * label-minimal subsystem of maximal probability is computed.
             * @return The index of the variable for the probability of the virtual initial state.
             */
            static std::pair<storm::expressions::Variable, uint_fast64_t> createVirtualInitialStateVariable(storm::solver::LpSolver& solver, bool maximizeProbability = false) {
                std::stringstream variableNameBuffer;
                variableNameBuffer << "pinit";
                storm::expressions::Variable variable = solver.addBoundedContinuousVariable(variableNameBuffer.str(), 0, 1);
                return std::make_pair(variable, 1);
            }
            
            /*!
             * Creates the variables for the problematic states in the model.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param stateInformation The information about the states in the model.
             * @return A mapping from problematic states to the index of the corresponding variables.
             */
            static std::pair<std::unordered_map<uint_fast64_t, storm::expressions::Variable>, uint_fast64_t> createProblematicStateVariables(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation) {
                std::stringstream variableNameBuffer;
                uint_fast64_t numberOfVariablesCreated = 0;
                std::unordered_map<uint_fast64_t, storm::expressions::Variable> resultingMap;
                
                for (auto state : stateInformation.problematicStates) {
                    // First check whether there is not already a variable for this state and advance to the next state
                    // in this case.
                    if (resultingMap.find(state) == resultingMap.end()) {
                        variableNameBuffer.str("");
                        variableNameBuffer.clear();
                        variableNameBuffer << "r" << state;
                        resultingMap[state] = solver.addBoundedContinuousVariable(variableNameBuffer.str(), 0, 1);
                        ++numberOfVariablesCreated;
                    }
                    
                    std::list<uint_fast64_t> const& relevantChoicesForState = choiceInformation.relevantChoicesForRelevantStates.at(state);
                    for (uint_fast64_t row : relevantChoicesForState) {
                        for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(row)) {
                            if (stateInformation.relevantStates.get(successorEntry.getColumn())) {
                                if (resultingMap.find(successorEntry.getColumn()) == resultingMap.end()) {
                                    variableNameBuffer.str("");
                                    variableNameBuffer.clear();
                                    variableNameBuffer << "r" << successorEntry.getColumn();
                                    resultingMap[successorEntry.getColumn()] = solver.addBoundedContinuousVariable(variableNameBuffer.str(), 0, 1);
                                    ++numberOfVariablesCreated;
                                }
                            }
                        }
                    }
                }
                return std::make_pair(resultingMap, numberOfVariablesCreated);
            }
            
            /*!
             * Creates the variables for the problematic choices in the model.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @return A mapping from problematic choices to the index of the corresponding variables.
             */
            static std::pair<std::unordered_map<std::pair<uint_fast64_t, uint_fast64_t>, storm::expressions::Variable, PairHash>, uint_fast64_t> createProblematicChoiceVariables(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation) {
                std::stringstream variableNameBuffer;
                uint_fast64_t numberOfVariablesCreated = 0;
                std::unordered_map<std::pair<uint_fast64_t, uint_fast64_t>, storm::expressions::Variable, PairHash> resultingMap;
                
                for (auto state : stateInformation.problematicStates) {
                    std::list<uint_fast64_t> const& relevantChoicesForState = choiceInformation.relevantChoicesForRelevantStates.at(state);
                    for (uint_fast64_t row : relevantChoicesForState) {
                        for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(row)) {
                            if (stateInformation.relevantStates.get(successorEntry.getColumn())) {
                                variableNameBuffer.str("");
                                variableNameBuffer.clear();
                                variableNameBuffer << "t" << state << "to" << successorEntry.getColumn();
                                resultingMap[std::make_pair(state, successorEntry.getColumn())] = solver.addBinaryVariable(variableNameBuffer.str());
                                ++numberOfVariablesCreated;
                            }
                        }
                    }
                }
                return std::make_pair(resultingMap, numberOfVariablesCreated);
            }
            
            /*!
             * Creates all variables needed to encode the problem as an MILP problem and returns a struct containing
             * information about the variables that were created. This implicitly establishes the objective function
             * passed to the solver.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             */
            static VariableInformation createVariables(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation) {
                // Create a struct that stores all information about variables.
                VariableInformation result;
                
                // Create variables for involved labels.
                std::pair<std::unordered_map<uint_fast64_t, storm::expressions::Variable>, uint_fast64_t> labelVariableResult = createLabelVariables(solver, choiceInformation.allRelevantLabels);
                result.labelToVariableMap = std::move(labelVariableResult.first);
                result.numberOfVariables += labelVariableResult.second;
                LOG4CPLUS_DEBUG(logger, "Created variables for labels.");
                
                // Create scheduler variables for relevant states and their actions.
                std::pair<std::unordered_map<uint_fast64_t, std::list<storm::expressions::Variable>>, uint_fast64_t> schedulerVariableResult = createSchedulerVariables(solver, stateInformation, choiceInformation);
                result.stateToChoiceVariablesMap = std::move(schedulerVariableResult.first);
                result.numberOfVariables += schedulerVariableResult.second;
                LOG4CPLUS_DEBUG(logger, "Created variables for nondeterministic choices.");

                // Create scheduler variables for nondeterministically choosing an initial state.
                std::pair<std::unordered_map<uint_fast64_t, storm::expressions::Variable>, uint_fast64_t> initialChoiceVariableResult = createInitialChoiceVariables(solver, labeledMdp, stateInformation);
                result.initialStateToChoiceVariableMap = std::move(initialChoiceVariableResult.first);
                result.numberOfVariables += initialChoiceVariableResult.second;
                LOG4CPLUS_DEBUG(logger, "Created variables for the nondeterministic choice of the initial state.");
                
                // Create variables for probabilities for all relevant states.
                std::pair<std::unordered_map<uint_fast64_t, storm::expressions::Variable>, uint_fast64_t> probabilityVariableResult = createProbabilityVariables(solver, stateInformation);
                result.stateToProbabilityVariableMap = std::move(probabilityVariableResult.first);
                result.numberOfVariables += probabilityVariableResult.second;
                LOG4CPLUS_DEBUG(logger, "Created variables for the reachability probabilities.");

                // Create a probability variable for a virtual initial state that nondeterministically chooses one of the system's real initial states as its target state.
                std::pair<storm::expressions::Variable, uint_fast64_t> virtualInitialStateVariableResult = createVirtualInitialStateVariable(solver);
                result.virtualInitialStateVariable = virtualInitialStateVariableResult.first;
                result.numberOfVariables += virtualInitialStateVariableResult.second;
                LOG4CPLUS_DEBUG(logger, "Created variables for the virtual initial state.");

                // Create variables for problematic states.
                std::pair<std::unordered_map<uint_fast64_t, storm::expressions::Variable>, uint_fast64_t> problematicStateVariableResult = createProblematicStateVariables(solver, labeledMdp, stateInformation, choiceInformation);
                result.problematicStateToVariableMap = std::move(problematicStateVariableResult.first);
                result.numberOfVariables += problematicStateVariableResult.second;
                LOG4CPLUS_DEBUG(logger, "Created variables for the problematic states.");

                // Create variables for problematic choices.
                std::pair<std::unordered_map<std::pair<uint_fast64_t, uint_fast64_t>, storm::expressions::Variable, PairHash>, uint_fast64_t> problematicTransitionVariableResult = createProblematicChoiceVariables(solver, labeledMdp, stateInformation, choiceInformation);
                result.problematicTransitionToVariableMap = problematicTransitionVariableResult.first;
                result.numberOfVariables += problematicTransitionVariableResult.second;
                LOG4CPLUS_DEBUG(logger, "Created variables for the problematic choices.");

                // Finally, we need to update the model to make the new variables usable.
                solver.update();
                LOG4CPLUS_INFO(logger, "Successfully created " << result.numberOfVariables << " MILP variables.");
                
                // Finally, return variable information struct.
                return result;
            }
            
            /*!
             * Asserts a constraint in the MILP problem that makes sure the reachability probability in the subsystem
             * exceeds the given threshold.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param variableInformation A struct with information about the variables of the model.
             * @param probabilityThreshold The probability that the subsystem must exceed.
             * @param strictBound A flag indicating whether the threshold must be exceeded or only matched.
             * @return The total number of constraints that were created.
             */
            static uint_fast64_t assertProbabilityGreaterThanThreshold(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, VariableInformation const& variableInformation, double probabilityThreshold, bool strictBound) {
                storm::expressions::Expression constraint;
                if (strictBound) {
                    constraint = variableInformation.virtualInitialStateVariable > solver.getConstant(probabilityThreshold);
                } else {
                    constraint = variableInformation.virtualInitialStateVariable >= solver.getConstant(probabilityThreshold);
                }
                solver.addConstraint("ProbGreaterThreshold", constraint);
                return 1;
            }
            
            /*!
             * Asserts constraints that make sure the selected policy is valid, i.e. chooses at most one action in each state.
             *
             * @param solver The MILP solver.
             * @param stateInformation The information about the states in the model.
             * @param variableInformation A struct with information about the variables of the model.
             * @return The total number of constraints that were created.
             */
            static uint_fast64_t assertValidPolicy(storm::solver::LpSolver& solver, StateInformation const& stateInformation, VariableInformation const& variableInformation) {
                // Assert that the policy chooses at most one action in each state of the system.
                uint_fast64_t numberOfConstraintsCreated = 0;
                for (auto state : stateInformation.relevantStates) {
                    std::list<storm::expressions::Variable> const& choiceVariableIndices = variableInformation.stateToChoiceVariablesMap.at(state);
                    storm::expressions::Expression constraint = solver.getConstant(0);

                    for (auto const& choiceVariable : choiceVariableIndices) {
                        constraint = constraint + choiceVariable;
                    }
                    
                    constraint = constraint <= solver.getConstant(1);
                    
                    solver.addConstraint("ValidPolicy" + std::to_string(numberOfConstraintsCreated), constraint);
                    ++numberOfConstraintsCreated;
                }
                
                // Now assert that the virtual initial state picks exactly one initial state from the system as its
                // successor state.
                storm::expressions::Expression constraint = solver.getConstant(0);
                for (auto const& initialStateVariablePair : variableInformation.initialStateToChoiceVariableMap) {
                    constraint = constraint + initialStateVariablePair.second;
                }
                constraint = constraint == solver.getConstant(1);
                
                solver.addConstraint("VirtualInitialStateChoosesOneInitialState", constraint);
                ++numberOfConstraintsCreated;
                
                return numberOfConstraintsCreated;
            }
            
            /*!
             * Asserts constraints that make sure the labels are included in the solution set if the policy selects a
             * choice that is labeled with the label in question.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @param variableInformation A struct with information about the variables of the model.
             * @return The total number of constraints that were created.
             */
            static uint_fast64_t assertChoicesImplyLabels(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation, VariableInformation const& variableInformation) {
                uint_fast64_t numberOfConstraintsCreated = 0;

                std::vector<boost::container::flat_set<uint_fast64_t>> const& choiceLabeling = labeledMdp.getChoiceLabeling();
                for (auto state : stateInformation.relevantStates) {
                    std::list<storm::expressions::Variable>::const_iterator choiceVariableIterator = variableInformation.stateToChoiceVariablesMap.at(state).begin();
                    for (auto choice : choiceInformation.relevantChoicesForRelevantStates.at(state)) {
                        for (auto label : choiceLabeling[choice]) {
                            storm::expressions::Expression constraint = variableInformation.labelToVariableMap.at(label) - *choiceVariableIterator >= solver.getConstant(0);
                            solver.addConstraint("ChoicesImplyLabels" + std::to_string(numberOfConstraintsCreated), constraint);
                            ++numberOfConstraintsCreated;
                        }
                        ++choiceVariableIterator;
                    }
                }
                return numberOfConstraintsCreated;
            }
            
            /*!
             * Asserts constraints that make sure the reachability probability is zero for states in which the policy
             * does not pick any outgoing action.
             *
             * @param solver The MILP solver.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @param variableInformation A struct with information about the variables of the model.
             * @return The total number of constraints that were created.
             */
            static uint_fast64_t assertZeroProbabilityWithoutChoice(storm::solver::LpSolver& solver, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation, VariableInformation const& variableInformation) {
                uint_fast64_t numberOfConstraintsCreated = 0;
                for (auto state : stateInformation.relevantStates) {
                    storm::expressions::Expression constraint = variableInformation.stateToProbabilityVariableMap.at(state);
                    for (auto const& choiceVariable : variableInformation.stateToChoiceVariablesMap.at(state)) {
                        constraint = constraint - choiceVariable;
                    }
                    constraint = constraint <= solver.getConstant(0);
                    solver.addConstraint("ProbabilityIsZeroIfNoAction" + std::to_string(numberOfConstraintsCreated), constraint);
                    ++numberOfConstraintsCreated;
                }
                return numberOfConstraintsCreated;
            }
            
            /*!
             * Asserts constraints that encode the correct reachability probabilities for all states.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param psiStates A bit vector characterizing the psi states in the model.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @param variableInformation A struct with information about the variables of the model.
             * @return The total number of constraints that were created.
             */
            static uint_fast64_t assertReachabilityProbabilities(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, storm::storage::BitVector const& psiStates, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation, VariableInformation const& variableInformation) {
                uint_fast64_t numberOfConstraintsCreated = 0;
                for (auto state : stateInformation.relevantStates) {
                    std::list<storm::expressions::Variable>::const_iterator choiceVariableIterator = variableInformation.stateToChoiceVariablesMap.at(state).begin();
                    for (auto choice : choiceInformation.relevantChoicesForRelevantStates.at(state)) {
                        storm::expressions::Expression constraint = variableInformation.stateToProbabilityVariableMap.at(state);
                        
                        double rightHandSide = 1;
                        for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(choice)) {
                            if (stateInformation.relevantStates.get(successorEntry.getColumn())) {
                                constraint = constraint - solver.getConstant(successorEntry.getValue()) * variableInformation.stateToProbabilityVariableMap.at(successorEntry.getColumn());
                            } else if (psiStates.get(successorEntry.getColumn())) {
                                rightHandSide += successorEntry.getValue();
                            }
                        }
                        
                        constraint = constraint + *choiceVariableIterator <= solver.getConstant(rightHandSide);
                        solver.addConstraint("ReachabilityProbabilities" + std::to_string(numberOfConstraintsCreated), constraint);
                        
                        ++numberOfConstraintsCreated;
                        ++choiceVariableIterator;
                    }
                }
                
                // Make sure that the virtual initial state is being assigned the probability from the initial state
                // that it selected as a successor state.
                for (auto const& initialStateVariablePair : variableInformation.initialStateToChoiceVariableMap) {
                    storm::expressions::Expression constraint = variableInformation.virtualInitialStateVariable - variableInformation.stateToProbabilityVariableMap.at(initialStateVariablePair.first) + initialStateVariablePair.second <= solver.getConstant(1);
                    solver.addConstraint("VirtualInitialStateHasCorrectProbability" + std::to_string(numberOfConstraintsCreated), constraint);
                    ++numberOfConstraintsCreated;
                }
                
                return numberOfConstraintsCreated;
            }
            
            /*!
             * Asserts constraints that make sure an unproblematic state is reachable from each problematic state.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @param variableInformation A struct with information about the variables of the model.
             * @return The total number of constraints that were created.
             */
            static uint_fast64_t assertUnproblematicStateReachable(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation, VariableInformation const& variableInformation) {
                uint_fast64_t numberOfConstraintsCreated = 0;

                for (auto stateListPair : choiceInformation.problematicChoicesForProblematicStates) {
                    for (auto problematicChoice : stateListPair.second) {
                        std::list<storm::expressions::Variable>::const_iterator choiceVariableIterator = variableInformation.stateToChoiceVariablesMap.at(stateListPair.first).begin();
                        for (auto relevantChoice : choiceInformation.relevantChoicesForRelevantStates.at(stateListPair.first)) {
                            if (relevantChoice == problematicChoice) {
                                break;
                            }
                            ++choiceVariableIterator;
                        }
                        
                        storm::expressions::Expression constraint = *choiceVariableIterator;
                        for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(problematicChoice)) {
                            constraint = constraint - variableInformation.problematicTransitionToVariableMap.at(std::make_pair(stateListPair.first, successorEntry.getColumn()));
                        }
                        constraint = constraint <= solver.getConstant(0);
                        
                        solver.addConstraint("UnproblematicStateReachable" + std::to_string(numberOfConstraintsCreated), constraint);
                        ++numberOfConstraintsCreated;
                    }
                }
                
                for (auto state : stateInformation.problematicStates) {
                    for (auto problematicChoice : choiceInformation.problematicChoicesForProblematicStates.at(state)) {
                        for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(problematicChoice)) {
                            storm::expressions::Expression constraint = variableInformation.problematicStateToVariableMap.at(state);
                            constraint = constraint - variableInformation.problematicStateToVariableMap.at(successorEntry.getColumn());
                            constraint = constraint + variableInformation.problematicTransitionToVariableMap.at(std::make_pair(state, successorEntry.getColumn()));
                            constraint = constraint < solver.getConstant(1);
                            
                            solver.addConstraint("UnproblematicStateReachable" + std::to_string(numberOfConstraintsCreated), constraint);
                            ++numberOfConstraintsCreated;
                        }
                    }
                }
                return numberOfConstraintsCreated;
            }
            
            /*
             * Asserts that labels that are on all paths from initial to target states are definitely taken.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param psiStates A bit vector characterizing the psi states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @param variableInformation A struct with information about the variables of the model.
             * @return The total number of constraints that were created.
             */
            static uint_fast64_t assertKnownLabels(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, storm::storage::BitVector const& psiStates, ChoiceInformation const& choiceInformation, VariableInformation const& variableInformation) {
                uint_fast64_t numberOfConstraintsCreated = 0;

                for (auto label : choiceInformation.knownLabels) {
                    storm::expressions::Expression constraint = variableInformation.labelToVariableMap.at(label) == solver.getConstant(1);
                    solver.addConstraint("KnownLabels" + std::to_string(numberOfConstraintsCreated), constraint);
                    ++numberOfConstraintsCreated;
                }

                return numberOfConstraintsCreated;
            }
            
            /*!
             * Asserts constraints that rule out many suboptimal policies.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param psiStates A bit vector characterizing the psi states in the model.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @param variableInformation A struct with information about the variables of the model.
             * @return The total number of constraints that were created.
             */
            static uint_fast64_t assertSchedulerCuts(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, storm::storage::BitVector const& psiStates, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation, VariableInformation const& variableInformation) {
                storm::storage::SparseMatrix<T> backwardTransitions = labeledMdp.getBackwardTransitions();
                uint_fast64_t numberOfConstraintsCreated = 0;
                
                for (auto state : stateInformation.relevantStates) {
                    // Assert that all states, that select an action, this action either has a non-zero probability to
                    // go to a psi state directly, or in the successor states, at least one action is selected as well.
                    std::list<storm::expressions::Variable>::const_iterator choiceVariableIterator = variableInformation.stateToChoiceVariablesMap.at(state).begin();
                    for (auto choice : choiceInformation.relevantChoicesForRelevantStates.at(state)) {
                        bool psiStateReachableInOneStep = false;
                        for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(choice)) {
                            if (psiStates.get(successorEntry.getColumn())) {
                                psiStateReachableInOneStep = true;
                            }
                        }
                        
                        if (!psiStateReachableInOneStep) {
                            storm::expressions::Expression constraint = *choiceVariableIterator;
                            for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(choice)) {
                                if (state != successorEntry.getColumn() && stateInformation.relevantStates.get(successorEntry.getColumn())) {
                                    std::list<storm::expressions::Variable> const& successorChoiceVariableIndices = variableInformation.stateToChoiceVariablesMap.at(successorEntry.getColumn());
                                    
                                    for (auto const& choiceVariable : successorChoiceVariableIndices) {
                                        constraint = constraint - choiceVariable;
                                    }
                                }
                            }
                            constraint = constraint <= solver.getConstant(1);
                            
                            solver.addConstraint("SchedulerCuts" + std::to_string(numberOfConstraintsCreated), constraint);
                            ++numberOfConstraintsCreated;
                        }
                        
                        ++choiceVariableIterator;
                    }
                    
                    // For all states assert that there is either a selected incoming transition in the subsystem or the
                    // state is the chosen initial state if there is one selected action in the current state.
                    storm::expressions::Expression constraint = solver.getConstant(0);

                    for (auto const& choiceVariable : variableInformation.stateToChoiceVariablesMap.at(state)) {
                        constraint = constraint + choiceVariable;
                    }
                    
                    // Compute the set of predecessors.
                    std::unordered_set<uint_fast64_t> predecessors;
                    for (auto const& predecessorEntry : backwardTransitions.getRow(state)) {
                        if (state != predecessorEntry.getColumn()) {
                            predecessors.insert(predecessorEntry.getColumn());
                        }
                    }
                    
                    for (auto predecessor : predecessors) {
                        // If the predecessor is not a relevant state, we need to skip it.
                        if (!stateInformation.relevantStates.get(predecessor)) {
                            continue;
                        }
                        
                        std::list<storm::expressions::Variable>::const_iterator choiceVariableIterator = variableInformation.stateToChoiceVariablesMap.at(predecessor).begin();
                        for (auto relevantChoice : choiceInformation.relevantChoicesForRelevantStates.at(predecessor)) {
                            bool choiceTargetsCurrentState = false;
                            
                            // Check if the current choice targets the current state.
                            for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(relevantChoice)) {
                                if (state == successorEntry.getColumn()) {
                                    choiceTargetsCurrentState = true;
                                    break;
                                }
                            }
                            
                            // If it does, we can add the choice to the sum.
                            if (choiceTargetsCurrentState) {
                                constraint = constraint - *choiceVariableIterator;
                            }
                            ++choiceVariableIterator;
                        }
                    }
                    
                    // If the current state is an initial state and is selected as a successor state by the virtual
                    // initial state, then this also justifies making a choice in the current state.
                    if (labeledMdp.getStates("init").get(state)) {
                        constraint = constraint - variableInformation.initialStateToChoiceVariableMap.at(state);
                    }
                    constraint = constraint <= solver.getConstant(0);
                    
                    solver.addConstraint("SchedulerCuts" + std::to_string(numberOfConstraintsCreated), constraint);
                    ++numberOfConstraintsCreated;
                }
                
                // Assert that at least one initial state selects at least one action.
                storm::expressions::Expression constraint = solver.getConstant(0);
                for (auto initialState : labeledMdp.getStates("init")) {
                    for (auto const& choiceVariable : variableInformation.stateToChoiceVariablesMap.at(initialState)) {
                        constraint = constraint + choiceVariable;
                    }
                }
                constraint = constraint >= solver.getConstant(1);
                solver.addConstraint("SchedulerCuts" + std::to_string(numberOfConstraintsCreated), constraint);
                ++numberOfConstraintsCreated;
                
                // Add constraints that ensure at least one choice is selected that targets a psi state.
                constraint = solver.getConstant(0);
                std::unordered_set<uint_fast64_t> predecessors;
                for (auto psiState : psiStates) {
                    // Compute the set of predecessors.
                    for (auto const& predecessorEntry : backwardTransitions.getRow(psiState)) {
                        if (psiState != predecessorEntry.getColumn()) {
                            predecessors.insert(predecessorEntry.getColumn());
                        }
                    }
                }
                
                for (auto predecessor : predecessors) {
                    // If the predecessor is not a relevant state, we need to skip it.
                    if (!stateInformation.relevantStates.get(predecessor)) {
                        continue;
                    }
                    
                    std::list<storm::expressions::Variable>::const_iterator choiceVariableIterator = variableInformation.stateToChoiceVariablesMap.at(predecessor).begin();
                    for (auto relevantChoice : choiceInformation.relevantChoicesForRelevantStates.at(predecessor)) {
                        bool choiceTargetsPsiState = false;
                        
                        // Check if the current choice targets the current state.
                        for (auto const& successorEntry : labeledMdp.getTransitionMatrix().getRow(relevantChoice)) {
                            if (psiStates.get(successorEntry.getColumn())) {
                                choiceTargetsPsiState = true;
                                break;
                            }
                        }
                        
                        // If it does, we can add the choice to the sum.
                        if (choiceTargetsPsiState) {
                            constraint = constraint + *choiceVariableIterator;
                        }
                        ++choiceVariableIterator;
                    }
                }
                constraint = constraint >= solver.getConstant(1);
                
                solver.addConstraint("SchedulerCuts" + std::to_string(numberOfConstraintsCreated), constraint);
                ++numberOfConstraintsCreated;
                
                return numberOfConstraintsCreated;
            }
            
            /*!
             * Builds a system of constraints that express that the reachability probability in the subsystem exceeeds
             * the given threshold.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param psiStates A bit vector characterizing all psi states in the model.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @param variableInformation A struct with information about the variables of the model.
             * @param probabilityThreshold The probability threshold the subsystem is required to exceed.
             * @param strictBound A flag indicating whether the threshold must be exceeded or only matched.
             * @param includeSchedulerCuts If set to true, additional constraints are asserted that reduce the set of
             * possible choices.
             */
            static void buildConstraintSystem(storm::solver::LpSolver& solver, storm::models::sparse::Mdp<T> const& labeledMdp, storm::storage::BitVector const& psiStates, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation, VariableInformation const& variableInformation, double probabilityThreshold, bool strictBound, bool includeSchedulerCuts = false) {
                // Assert that the reachability probability in the subsystem exceeds the given threshold.
                uint_fast64_t numberOfConstraints = assertProbabilityGreaterThanThreshold(solver, labeledMdp, variableInformation, probabilityThreshold, strictBound);
                LOG4CPLUS_DEBUG(logger, "Asserted that reachability probability exceeds threshold.");

                // Add constraints that assert the policy takes at most one action in each state.
                numberOfConstraints += assertValidPolicy(solver, stateInformation, variableInformation);
                LOG4CPLUS_DEBUG(logger, "Asserted that policy is valid.");

                // Add constraints that assert the labels that belong to some taken choices are taken as well.
                numberOfConstraints += assertChoicesImplyLabels(solver, labeledMdp, stateInformation, choiceInformation, variableInformation);
                LOG4CPLUS_DEBUG(logger, "Asserted that labels implied by choices are taken.");

                // Add constraints that encode that the reachability probability from states which do not pick any action
                // is zero.
                numberOfConstraints += assertZeroProbabilityWithoutChoice(solver, stateInformation, choiceInformation, variableInformation);
                LOG4CPLUS_DEBUG(logger, "Asserted that reachability probability is zero if no choice is taken.");

                // Add constraints that encode the reachability probabilities for states.
                numberOfConstraints += assertReachabilityProbabilities(solver, labeledMdp, psiStates, stateInformation, choiceInformation, variableInformation);
                LOG4CPLUS_DEBUG(logger, "Asserted constraints for reachability probabilities.");

                // Add constraints that ensure the reachability of an unproblematic state from each problematic state.
                numberOfConstraints += assertUnproblematicStateReachable(solver, labeledMdp, stateInformation, choiceInformation, variableInformation);
                LOG4CPLUS_DEBUG(logger, "Asserted that unproblematic state reachable from problematic states.");

                // Add constraints that express that certain labels are already known to be taken.
                numberOfConstraints += assertKnownLabels(solver, labeledMdp, psiStates, choiceInformation, variableInformation);
                LOG4CPLUS_DEBUG(logger, "Asserted known labels are taken.");
                
                // If required, assert additional constraints that reduce the number of possible policies.
                if (includeSchedulerCuts) {
                    numberOfConstraints += assertSchedulerCuts(solver, labeledMdp, psiStates, stateInformation, choiceInformation, variableInformation);
                    LOG4CPLUS_DEBUG(logger, "Asserted scheduler cuts.");
                }
                
                // Finally, we can tell the solver to incorporate the latest changes.
                solver.update();
                
                LOG4CPLUS_INFO(logger, "Successfully created " << numberOfConstraints << " MILP constraints.");
            }
            
            /*!
             * Computes the set of labels that was used in the given optimized model.
             *
             * @param solver The MILP solver.
             * @param variableInformation A struct with information about the variables of the model.
             */
            static boost::container::flat_set<uint_fast64_t> getUsedLabelsInSolution(storm::solver::LpSolver const& solver, VariableInformation const& variableInformation) {
                boost::container::flat_set<uint_fast64_t> result;

                for (auto const& labelVariablePair : variableInformation.labelToVariableMap) {
                    bool labelTaken = solver.getBinaryValue(labelVariablePair.second);
                    
                    if (labelTaken) {
                        result.insert(labelVariablePair.first);
                    }
                }
                return result;
            }
            
            /*!
             * Computes a mapping from relevant states to choices such that a state is mapped to one of its choices if
             * it is selected by the subsystem computed by the solver.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param stateInformation The information about the states in the model.
             * @param choiceInformation The information about the choices in the model.
             * @param variableInformation A struct with information about the variables of the model.
             */
            static std::map<uint_fast64_t, uint_fast64_t> getChoices(storm::solver::LpSolver const& solver, storm::models::sparse::Mdp<T> const& labeledMdp, StateInformation const& stateInformation, ChoiceInformation const& choiceInformation, VariableInformation const& variableInformation) {
                std::map<uint_fast64_t, uint_fast64_t> result;
                
                for (auto state : stateInformation.relevantStates) {
                    std::list<storm::expressions::Variable>::const_iterator choiceVariableIterator = variableInformation.stateToChoiceVariablesIndexMap.at(state).begin();
                    for (auto choice : choiceInformation.relevantChoicesForRelevantStates.at(state)) {
                        bool choiceTaken = solver.getBinaryValue(*choiceVariableIterator);
                        ++choiceVariableIterator;
                        if (choiceTaken) {
                            result.emplace_hint(result.end(), state, choice);
                        }
                    }
                }
                
                return result;
            }
            
            /*!
             * Computes the reachability probability and the selected initial state in the given optimized MILP model.
             *
             * @param solver The MILP solver.
             * @param labeledMdp The labeled MDP.
             * @param variableInformation A struct with information about the variables of the model.
             */
            static std::pair<uint_fast64_t, double> getReachabilityProbability(storm::solver::LpSolver const& solver, storm::models::sparse::Mdp<T> const& labeledMdp, VariableInformation const& variableInformation) {
                uint_fast64_t selectedInitialState = 0;
                for (auto const& initialStateVariablePair : variableInformation.initialStateToChoiceVariableMap) {
                    bool initialStateChosen = solver.getBinaryValue(initialStateVariablePair.second);
                    if (initialStateChosen) {
                        selectedInitialState = initialStateVariablePair.first;
                        break;
                    }
                }
                
                double reachabilityProbability = solver.getContinuousValue(variableInformation.virtualInitialStateVariable);
                return std::make_pair(selectedInitialState, reachabilityProbability);
            }
                
        public:

            static boost::container::flat_set<uint_fast64_t> getMinimalLabelSet(storm::logic::Formula const& pathFormula, storm::models::sparse::Mdp<T> const& labeledMdp, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, double probabilityThreshold, bool strictBound, bool checkThresholdFeasible = false, bool includeSchedulerCuts = false) {
                // (0) Check whether the MDP is indeed labeled.
                if (!labeledMdp.hasChoiceLabeling()) {
                    throw storm::exceptions::InvalidArgumentException() << "Minimal label set generation is impossible for unlabeled model.";
                }
                
                // (1) Check whether its possible to exceed the threshold if checkThresholdFeasible is set.
                double maximalReachabilityProbability = 0;
                if (checkThresholdFeasible) {
                    storm::modelchecker::SparseMdpPrctlModelChecker<T> modelchecker(labeledMdp);
                    std::vector<T> result = modelchecker.computeUntilProbabilitiesHelper(false, phiStates, psiStates, false);
                    for (auto state : labeledMdp.getInitialStates()) {
                        maximalReachabilityProbability = std::max(maximalReachabilityProbability, result[state]);
                    }
                    STORM_LOG_THROW((strictBound && maximalReachabilityProbability >= probabilityThreshold) || (!strictBound && maximalReachabilityProbability > probabilityThreshold), storm::exceptions::InvalidArgumentException, "Given probability threshold " << probabilityThreshold << " can not be " << (strictBound ? "achieved" : "exceeded") << " in model with maximal reachability probability of " << maximalReachabilityProbability << ".");
                    std::cout << std::endl << "Maximal reachability in model is " << maximalReachabilityProbability << "." << std::endl << std::endl;
                }

                // (2) Identify relevant and problematic states.
                StateInformation stateInformation = determineRelevantAndProblematicStates(labeledMdp, phiStates, psiStates);
                
                // (3) Determine sets of relevant labels and problematic choices.
                ChoiceInformation choiceInformation = determineRelevantAndProblematicChoices(labeledMdp, stateInformation, psiStates);
                
                // (4) Encode resulting system as MILP problem.
                std::shared_ptr<storm::solver::LpSolver> solver = storm::utility::solver::getLpSolver("MinimalCommandSetCounterexample");
                
                //  (4.1) Create variables.
                VariableInformation variableInformation = createVariables(*solver, labeledMdp, stateInformation, choiceInformation);
 
                //  (4.2) Construct constraint system.
                buildConstraintSystem(*solver, labeledMdp, psiStates, stateInformation, choiceInformation, variableInformation, probabilityThreshold, strictBound, includeSchedulerCuts);
                
                solver->writeModelToFile("model.lp");
                
                // (4.3) Optimize the model.
                solver->optimize();
                
                // (4.4) Read off result from variables.
                boost::container::flat_set<uint_fast64_t> usedLabelSet = getUsedLabelsInSolution(*solver, variableInformation);
                usedLabelSet.insert(choiceInformation.knownLabels.begin(), choiceInformation.knownLabels.end());
                
                // (5) Return result.
                return usedLabelSet;
            }
            
            /*!
             * Computes a (minimally labeled) counterexample for the given model and (safety) formula. If the model satisfies the property, an exception is thrown.
             *
             * @param labeledMdp A labeled MDP that is the model in which to generate the counterexample.
             * @param formulaPtr A pointer to a safety formula. The outermost operator must be a probabilistic bound operator with a strict upper bound. The nested
             * formula can be either an unbounded until formula or an eventually formula.
             */
            static void computeCounterexample(storm::prism::Program const& program, storm::models::sparse::Mdp<T> const& labeledMdp, std::shared_ptr<storm::logic::Formula> const& formula) {
                std::cout << std::endl << "Generating minimal label counterexample for formula " << *formula << std::endl;
                
                STORM_LOG_THROW(formula->isProbabilityOperatorFormula(), storm::exceptions::InvalidPropertyException, "Counterexample generation does not support this kind of formula. Expecting a probability operator as the outermost formula element.");
                storm::logic::ProbabilityOperatorFormula const& probabilityOperator = formula->asProbabilityOperatorFormula();
                STORM_LOG_THROW(probabilityOperator.hasBound(), storm::exceptions::InvalidPropertyException, "Counterexample generation only supports bounded formulas.");
                storm::logic::ComparisonType comparisonType = probabilityOperator.getComparisonType();
                STORM_LOG_THROW(comparisonType == storm::logic::ComparisonType::Less || comparisonType == storm::logic::ComparisonType::LessEqual, storm::exceptions::InvalidPropertyException, "Counterexample generation only supports formulas with an upper probability bound.");
                STORM_LOG_THROW(probabilityOperator.getSubformula().isUntilFormula() || probabilityOperator.getSubformula().isEventuallyFormula(), storm::exceptions::InvalidPropertyException, "Path formula is required to be of the form 'phi U psi' for counterexample generation.");
                
                bool strictBound = comparisonType == storm::logic::ComparisonType::Less;
                double bound = probabilityOperator.getBound();

                storm::storage::BitVector phiStates;
                storm::storage::BitVector psiStates;
                storm::modelchecker::SparseMdpPrctlModelChecker<T> modelchecker(labeledMdp);
                
                if (probabilityOperator.getSubformula().isUntilFormula()) {
                    storm::logic::UntilFormula const& untilFormula = probabilityOperator.getSubformula().asUntilFormula();
                    
                    std::unique_ptr<storm::modelchecker::CheckResult> leftResult = modelchecker.check(untilFormula.getLeftSubformula());
                    std::unique_ptr<storm::modelchecker::CheckResult> rightResult = modelchecker.check(untilFormula.getRightSubformula());
                    
                    storm::modelchecker::ExplicitQualitativeCheckResult const& leftQualitativeResult = leftResult->asExplicitQualitativeCheckResult();
                    storm::modelchecker::ExplicitQualitativeCheckResult const& rightQualitativeResult = rightResult->asExplicitQualitativeCheckResult();
                    
                    phiStates = leftQualitativeResult.getTruthValuesVector();
                    psiStates = rightQualitativeResult.getTruthValuesVector();
                } else if (probabilityOperator.getSubformula().isEventuallyFormula()) {
                    storm::logic::EventuallyFormula const& eventuallyFormula = probabilityOperator.getSubformula().asEventuallyFormula();
                    
                    std::unique_ptr<storm::modelchecker::CheckResult> subResult = modelchecker.check(eventuallyFormula.getSubformula());
                    
                    storm::modelchecker::ExplicitQualitativeCheckResult const& subQualitativeResult = subResult->asExplicitQualitativeCheckResult();
                    
                    phiStates = storm::storage::BitVector(labeledMdp.getNumberOfStates(), true);
                    psiStates = subQualitativeResult.getTruthValuesVector();
                }
                
                // Delegate the actual computation work to the function of equal name.
                auto startTime = std::chrono::high_resolution_clock::now();
                boost::container::flat_set<uint_fast64_t> usedLabelSet = getMinimalLabelSet(probabilityOperator.getSubformula(), labeledMdp, phiStates, psiStates, bound, strictBound, true, storm::settings::counterexampleGeneratorSettings().isUseSchedulerCutsSet());
                auto endTime = std::chrono::high_resolution_clock::now();
                std::cout << std::endl << "Computed minimal label set of size " << usedLabelSet.size() << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << "ms." << std::endl;

                std::cout << "Resulting program:" << std::endl;
                storm::prism::Program restrictedProgram = program.restrictCommands(usedLabelSet);
                std::cout << restrictedProgram << std::endl;
                std::cout << std::endl << "-------------------------------------------" << std::endl;
            }
            
        };
        
    } // namespace counterexamples
} // namespace storm

#endif /* STORM_COUNTEREXAMPLES_MILPMINIMALLABELSETGENERATOR_MDP_H_ */
