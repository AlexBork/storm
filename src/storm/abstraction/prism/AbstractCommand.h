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
    
    namespace prism {
        // Forward-declare concrete command and assignment classes.
        class Command;
        class Assignment;
    }
    
    namespace abstraction {
        template <storm::dd::DdType DdType>
        class AbstractionInformation;

        template <storm::dd::DdType DdType>
        class BottomStateResult;

        template<storm::dd::DdType DdType>
        struct GameBddResult;

        namespace prism {
            template <storm::dd::DdType DdType, typename ValueType>
            class AbstractCommand {
            public:
                /*!
                 * Constructs an abstract command from the given command and the initial predicates.
                 *
                 * @param command The concrete command for which to build the abstraction.
                 * @param abstractionInformation An object holding information about the abstraction such as predicates and BDDs.
                 * @param smtSolverFactory A factory that is to be used for creating new SMT solvers.
                 * @param guardIsPredicate A flag indicating whether the guard of the command was added as a predicate.
                 */
                AbstractCommand(storm::prism::Command const& command, AbstractionInformation<DdType>& abstractionInformation, std::shared_ptr<storm::utility::solver::SmtSolverFactory> const& smtSolverFactory, bool guardIsPredicate = false);
                               
                /*!
                 * Refines the abstract command with the given predicates.
                 *
                 * @param predicates The new predicates.
                 * @param forceRecomputation If set, the BDD is recomputed even if the relevant predicates have not changed.
                 */
                void refine(std::vector<uint_fast64_t> const& predicates, bool forceRecomputation = false);
                
                /*!
                 * Computes the abstraction of the command wrt. to the current set of predicates.
                 *
                 * @return The abstraction of the command in the form of a BDD together with the number of DD variables
                 * used to encode the choices of player 2.
                 */
                GameBddResult<DdType> getAbstractBdd();
                
                /*!
                 * Retrieves the transitions to bottom states of this command.
                 *
                 * @param reachableStates A BDD representing the reachable states.
                 * @param numberOfPlayer2Variables The number of variables used to encode the choices of player 2.
                 * @return The bottom state transitions in the form of a BDD.
                 */
                BottomStateResult<DdType> getBottomStateTransitions(storm::dd::Bdd<DdType> const& reachableStates, uint_fast64_t numberOfPlayer2Variables);

                /*!
                 * Retrieves an ADD that maps the encoding of the command and its updates to their probabilities.
                 *
                 * @return The command-update probability ADD.
                 */
                storm::dd::Add<DdType, ValueType> getCommandUpdateProbabilitiesAdd() const;
                
                /*!
                 * Retrieves the concrete command that is abstracted by this abstract command.
                 *
                 * @return The concrete command.
                 */
                storm::prism::Command const& getConcreteCommand() const;
                
                /*!
                 * Notifies this abstract command that its guard is now a predicate. This affects the computation of the
                 * bottom states.
                 */
                void notifyGuardIsPredicate();
                
            private:
                /*!
                 * Determines the relevant predicates for source as well as successor states wrt. to the given assignments
                 * (that, for example, form an update).
                 *
                 * @param assignments The assignments that are to be considered.
                 * @return A pair whose first component represents the relevant source predicates and whose second
                 * component represents the relevant successor state predicates.
                 */
                std::pair<std::set<uint_fast64_t>, std::set<uint_fast64_t>> computeRelevantPredicates(std::vector<storm::prism::Assignment> const& assignments) const;
                
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
                
                // The concrete command this abstract command refers to.
                std::reference_wrapper<storm::prism::Command const> command;
                
                // The local expression-related information.
                LocalExpressionInformation localExpressionInformation;
                
                // The evaluator used to translate the probability expressions.
                storm::expressions::ExpressionEvaluator<ValueType> evaluator;
                
                // The currently relevant source/successor predicates and the corresponding variables.
                std::pair<std::vector<std::pair<storm::expressions::Variable, uint_fast64_t>>, std::vector<std::vector<std::pair<storm::expressions::Variable, uint_fast64_t>>>> relevantPredicatesAndVariables;
                
                // The most recent result of a call to computeDd. If nothing has changed regarding the relevant
                // predicates, this result may be reused.
                GameBddResult<DdType> cachedDd;
                
                // All relevant decision variables over which to perform AllSat.
                std::vector<storm::expressions::Variable> decisionVariables;
                
                // A flag indicating whether the guard of the command was added as a predicate. If this is true, there
                // is no need to compute bottom states.
                bool guardIsPredicate;
                
                // The abstract guard of the command. This is only used if the guard is not a predicate, because it can
                // then be used to constrain the bottom state abstractor.
                storm::dd::Bdd<DdType> abstractGuard;
                
                // A state-set abstractor used to determine the bottom states if not all guards were added.
                StateSetAbstractor<DdType, ValueType> bottomStateAbstractor;
            };
        }
    }
}
