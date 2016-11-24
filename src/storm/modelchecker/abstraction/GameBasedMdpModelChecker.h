#ifndef STORM_MODELCHECKER_GAMEBASEDMDPMODELCHECKER_H_
#define STORM_MODELCHECKER_GAMEBASEDMDPMODELCHECKER_H_

#include "storm/modelchecker/AbstractModelChecker.h"

#include "storm/storage/prism/Program.h"

#include "storm/storage/dd/DdType.h"

#include "storm/storage/SymbolicModelDescription.h"

#include "storm/logic/Bound.h"

#include "storm/utility/solver.h"
#include "storm/utility/graph.h"

namespace storm {
    namespace abstraction {
        template<storm::dd::DdType Type, typename ValueType>
        class MenuGame;
    }
    
    namespace modelchecker {
        namespace detail {
            template<storm::dd::DdType Type>
            struct GameProb01ResultMinMax {
            public:
                GameProb01ResultMinMax() = default;

                storm::utility::graph::GameProb01Result<Type> prob0Min;
                storm::utility::graph::GameProb01Result<Type> prob1Min;
                storm::utility::graph::GameProb01Result<Type> prob0Max;
                storm::utility::graph::GameProb01Result<Type> prob1Max;
            };
        }

        template<storm::dd::DdType Type, typename ModelType>
        class GameBasedMdpModelChecker : public AbstractModelChecker<ModelType> {
        public:
            typedef typename ModelType::ValueType ValueType;
            
            /*!
             * Constructs a model checker whose underlying model is implicitly given by the provided program. All
             * verification calls will be answererd with respect to this model.
             *
             * @param model The model description that (symbolically) specifies the model to check.
             * @param smtSolverFactory A factory used to create SMT solver when necessary.
             */
            explicit GameBasedMdpModelChecker(storm::storage::SymbolicModelDescription const& model, std::shared_ptr<storm::utility::solver::SmtSolverFactory> const& smtSolverFactory = std::make_shared<storm::utility::solver::MathsatSmtSolverFactory>());
            
            /// Overridden methods from super class.
            virtual bool canHandle(CheckTask<storm::logic::Formula> const& checkTask) const override;
            virtual std::unique_ptr<CheckResult> computeUntilProbabilities(CheckTask<storm::logic::UntilFormula> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeReachabilityProbabilities(CheckTask<storm::logic::EventuallyFormula> const& checkTask) override;
            
        private:
            /*!
             * Performs the core part of the abstraction-refinement loop.
             */
            std::unique_ptr<CheckResult> performGameBasedAbstractionRefinement(CheckTask<storm::logic::Formula> const& checkTask, storm::expressions::Expression const& constraintExpression, storm::expressions::Expression const& targetStateExpression);
            
            /*!
             * Retrieves the initial predicates for the abstraction.
             */
            std::vector<storm::expressions::Expression> getInitialPredicates(storm::expressions::Expression const& constraintExpression, storm::expressions::Expression const& targetStateExpression);
            
            /*!
             * Derives the optimization direction of player 1.
             */
            storm::OptimizationDirection getPlayer1Direction(CheckTask<storm::logic::Formula> const& checkTask);
            
            /*!
             * Performs a qualitative check on the the given game to compute the (player 1) states that have probability
             * 0 or 1, respectively, to reach a target state and only visiting constraint states before.
             */
            std::unique_ptr<CheckResult> computeProb01States(CheckTask<storm::logic::Formula> const& checkTask, detail::GameProb01ResultMinMax<Type>& qualitativeResult, storm::abstraction::MenuGame<Type, ValueType> const& game, storm::OptimizationDirection player1Direction, storm::dd::Bdd<Type> const& transitionMatrixBdd, storm::dd::Bdd<Type> const& initialStates, storm::dd::Bdd<Type> const& constraintStates, storm::dd::Bdd<Type> const& targetStates);
            storm::utility::graph::GameProb01Result<Type> computeProb01States(bool prob0, storm::OptimizationDirection player1Direction, storm::OptimizationDirection player2Direction, storm::abstraction::MenuGame<Type, ValueType> const& game, storm::dd::Bdd<Type> const& transitionMatrixBdd, storm::dd::Bdd<Type> const& constraintStates, storm::dd::Bdd<Type> const& targetStates);
            
            /*
             * Retrieves the expression characterized by the formula. The formula needs to be propositional.
             */
            storm::expressions::Expression getExpression(storm::logic::Formula const& formula);
            
            /// The preprocessed model that contains only one module/automaton and otherwhise corresponds to the semantics
            /// of the original model description.
            storm::storage::SymbolicModelDescription preprocessedModel;
            
            /// A factory that is used for creating SMT solvers when needed.
            std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory;
        };
    }
}

#endif /* STORM_MODELCHECKER_GAMEBASEDMDPMODELCHECKER_H_ */
