#pragma once

#include <memory>
#include <vector>
#include <set>
#include <map>

#include "storm/abstraction/LocalExpressionInformation.h"
#include "storm/abstraction/StateSetAbstractor.h"
#include "storm/abstraction/GameBddResult.h"

#include "storm/storage/expressions/ExpressionEvaluator.h"

#include "storm/storage/dd/DdType.h"
#include "storm/storage/expressions/Expression.h"

#include "storm/solver/SmtSolver.h"

namespace storm {
    namespace utility {
        namespace solver {
            class SmtSolverFactory;
        }
    }
    
    namespace dd {
        template <storm::dd::DdType DdType>
        class Bdd;

        template <storm::dd::DdType DdType, typename ValueType>
        class Add;
    }
    
    namespace jani {
        // Forward-declare concrete edge and assignment classes.
        class Edge;
        class Assignment;
        class OrderedAssignments;
    }
    
    namespace abstraction {
        template <storm::dd::DdType DdType>
        class AbstractionInformation;

        template <storm::dd::DdType DdType>
        class BottomStateResult;

        namespace jani {
            template <storm::dd::DdType DdType, typename ValueType>
            class EdgeAbstractor {
            public:
                /*!
                 * Constructs an abstract edge from the given command and the initial predicates.
                 *
                 * @param edgeId The ID to assign to the edge.
                 * @param edge The concrete edge for which to build the abstraction.
                 * @param abstractionInformation An object holding information about the abstraction such as predicates and BDDs.
                 * @param smtSolverFactory A factory that is to be used for creating new SMT solvers.
                 * @param allowInvalidSuccessors A flag indicating whether it is allowed to enumerate invalid successors.
                 */
                EdgeAbstractor(uint64_t edgeId, storm::jani::Edge const& edge, AbstractionInformation<DdType>& abstractionInformation, std::shared_ptr<storm::utility::solver::SmtSolverFactory> const& smtSolverFactory, bool allowInvalidSuccessors);
                               
                /*!
                 * Refines the abstract edge with the given predicates.
                 *
                 * @param predicates The new predicates.
                 */
                void refine(std::vector<uint_fast64_t> const& predicates);
                
                /*!
                 * Retrieves the guard of this edge.
                 */
                storm::expressions::Expression const& getGuard() const;
                
                /*!
                 * Retrieves a mapping from variables to expressions that define their updates wrt. to the given
                 * auxiliary choice.
                 */
                std::map<storm::expressions::Variable, storm::expressions::Expression> getVariableUpdates(uint64_t auxiliaryChoice) const;
                
                /*!
                 * Computes the abstraction of the edge wrt. to the current set of predicates.
                 *
                 * @return The abstraction of the edge in the form of a BDD together with the number of DD variables
                 * used to encode the choices of player 2.
                 */
                GameBddResult<DdType> abstract();
                
                /*!
                 * Retrieves the transitions to bottom states of this edge.
                 *
                 * @param reachableStates A BDD representing the reachable states.
                 * @param numberOfPlayer2Variables The number of variables used to encode the choices of player 2.
                 * @return The bottom state transitions in the form of a BDD.
                 */
                BottomStateResult<DdType> getBottomStateTransitions(storm::dd::Bdd<DdType> const& reachableStates, uint_fast64_t numberOfPlayer2Variables);

                /*!
                 * Retrieves an ADD that maps the encoding of the edge and its updates to their probabilities.
                 *
                 * @return The edge-update probability ADD.
                 */
                storm::dd::Add<DdType, ValueType> getEdgeUpdateProbabilitiesAdd() const;
                
                /*!
                 * Retrieves the concrete edge that is abstracted by this abstract edge.
                 *
                 * @return The concrete edge.
                 */
                storm::jani::Edge const& getConcreteEdge() const;
                
            private:
                /*!
                 * Determines the relevant predicates for source as well as successor states wrt. to the given assignments
                 * (that, for example, form an update).
                 *
                 * @param assignments The assignments that are to be considered.
                 * @return A pair whose first component represents the relevant source predicates and whose second
                 * component represents the relevant successor state predicates.
                 */
                std::pair<std::set<uint_fast64_t>, std::set<uint_fast64_t>> computeRelevantPredicates(storm::jani::OrderedAssignments const& assignments) const;
                
                /*!
                 * Determines the relevant predicates for source as well as successor states.
                 *
                 * @return A pair whose first component represents the relevant source predicates and whose second
                 * component represents the relevant successor state predicates.
                 */
                std::pair<std::set<uint_fast64_t>, std::vector<std::set<uint_fast64_t>>> computeRelevantPredicates() const;

                /*!
                 * Checks whether the relevant predicates changed.
                 *
                 * @param newRelevantPredicates The new relevant predicates.
                 */
                bool relevantPredicatesChanged(std::pair<std::set<uint_fast64_t>, std::vector<std::set<uint_fast64_t>>> const& newRelevantPredicates) const;
                                
                /*!
                 * Takes the new relevant predicates and creates the appropriate variables and assertions for the ones
                 * that are currently missing.
                 *
                 * @param newRelevantPredicates The new relevant predicates.
                 */
                void addMissingPredicates(std::pair<std::set<uint_fast64_t>, std::vector<std::set<uint_fast64_t>>> const& newRelevantPredicates);
                
                /*!
                 * Translates the given model to a source state DD.
                 *
                 * @param model The model to translate.
                 * @return The source state encoded as a DD.
                 */
                storm::dd::Bdd<DdType> getSourceStateBdd(storm::solver::SmtSolver::ModelReference const& model) const;

                /*!
                 * Translates the given model to a distribution over successor states.
                 *
                 * @param model The model to translate.
                 * @return The source state encoded as a DD.
                 */
                storm::dd::Bdd<DdType> getDistributionBdd(storm::solver::SmtSolver::ModelReference const& model) const;
                
                /*!
                 * Recomputes the cached BDD. This needs to be triggered if any relevant predicates change.
                 */
                void recomputeCachedBdd();

                /*!
                 * Computes the missing state identities.
                 *
                 * @return A BDD that represents the all missing state identities.
                 */
                storm::dd::Bdd<DdType> computeMissingIdentities() const;
                
                /*!
                 * Computes the missing state identities for the updates.
                 *
                 * @return A BDD that represents the state identities for predicates that are irrelevant for the
                 * successor states.
                 */
                storm::dd::Bdd<DdType> computeMissingUpdateIdentities() const;
                
                /*!
                 * Retrieves the abstraction information object.
                 *
                 * @return The abstraction information object.
                 */
                AbstractionInformation<DdType> const& getAbstractionInformation() const;

                /*!
                 * Retrieves the abstraction information object.
                 *
                 * @return The abstraction information object.
                 */
                AbstractionInformation<DdType>& getAbstractionInformation();
                
                /*!
                 * Computes the globally missing state identities.
                 *
                 * @return A BDD that represents the global state identities for predicates that are irrelevant for the
                 * source and successor states.
                 */
                storm::dd::Bdd<DdType> computeMissingGlobalIdentities() const;
                
                // An SMT responsible for this abstract command.
                std::unique_ptr<storm::solver::SmtSolver> smtSolver;

                // The abstraction-related information.
                std::reference_wrapper<AbstractionInformation<DdType>> abstractionInformation;
                
                // The ID of the edge.
                uint64_t edgeId;
                
                // The concrete edge this abstract command refers to.
                std::reference_wrapper<storm::jani::Edge const> edge;
                
                // The local expression-related information.
                LocalExpressionInformation<DdType> localExpressionInformation;
                
                // The evaluator used to translate the probability expressions.
                storm::expressions::ExpressionEvaluator<ValueType> evaluator;
                
                // The currently relevant source/successor predicates and the corresponding variables.
                std::pair<std::vector<std::pair<storm::expressions::Variable, uint_fast64_t>>, std::vector<std::vector<std::pair<storm::expressions::Variable, uint_fast64_t>>>> relevantPredicatesAndVariables;
                
                // The set of all relevant predicates.
                std::set<uint64_t> allRelevantPredicates;
                
                // The most recent result of a call to computeDd. If nothing has changed regarding the relevant
                // predicates, this result may be reused.
                GameBddResult<DdType> cachedDd;
                
                // All relevant decision variables over which to perform AllSat.
                std::vector<storm::expressions::Variable> decisionVariables;
                
                // A flag indicating whether it is allowed to enumerate invalid successors. Invalid successors may be
                // enumerated if the predicates that are (indirectly) related to an assignment variable are not
                // considered as source predicates.
                bool allowInvalidSuccessors;
                
                // A flag indicating whether the computation of bottom states can be skipped (for example, if the bottom
                // states become empty at some point).
                bool skipBottomStates;
                
                // A flag remembering whether we need to force recomputation of the BDD.
                bool forceRecomputation;
                
                // The abstract guard of the edge. This is only used if the guard is not a predicate, because it can
                // then be used to constrain the bottom state abstractor.
                storm::dd::Bdd<DdType> abstractGuard;
                
                // A state-set abstractor used to determine the bottom states if not all guards were added.
                StateSetAbstractor<DdType, ValueType> bottomStateAbstractor;
            };
        }
    }
}
