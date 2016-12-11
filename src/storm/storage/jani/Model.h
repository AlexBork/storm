#pragma once

#include <memory>

#include "storm/storage/jani/Action.h"
#include "storm/storage/jani/ModelType.h"
#include "storm/storage/jani/Automaton.h"
#include "storm/storage/jani/Constant.h"
#include "storm/storage/jani/Composition.h"

#include "storm/utility/solver.h"
#include "storm/utility/vector.h"

namespace storm {
    namespace expressions {
        class ExpressionManager;
    }
    
    namespace jani {
        
        class Exporter;
        class SynchronizationVector;
        
        class Model {
        public:
            friend class Exporter;
            
            /*!
             * Creates an uninitialized model.
             */
            Model();
            
            /*!
             * Creates an empty model with the given type.
             */
            Model(std::string const& name, ModelType const& modelType, uint64_t version = 1, boost::optional<std::shared_ptr<storm::expressions::ExpressionManager>> const& expressionManager = boost::none);

            /*!
             * Retrieves the expression manager responsible for the expressions in the model.
             */
            storm::expressions::ExpressionManager& getManager() const;
            
            /*!
             * Retrieves the JANI-version of the model.
             */
            uint64_t getJaniVersion() const;
            
            /*!
             * Retrieves the type of the model.
             */
            ModelType const& getModelType() const;
            
            /*!
             * Retrievest the name of the model.
             */
            std::string const& getName() const;
            
            /*!
             * Flatten the composition to obtain an equivalent model that contains exactly one automaton that has the
             * standard composition.
             *
             * @param smtSolverFactory A factory that can be used to create new SMT solvers.
             */
            Model flattenComposition(std::shared_ptr<storm::utility::solver::SmtSolverFactory> const& smtSolverFactory = std::make_shared<storm::utility::solver::SmtSolverFactory>()) const;
            
            /**
             * Checks whether the model has an action with the given name.
             *
             * @param name The name to look for.
             */
            bool hasAction(std::string const& name) const;
            
            /**
             * Get the index of the action
             * @param the name of the model
             * @return the index of the (unique) action with the given name, undefined if no such action is present.
             */
            uint64_t getActionIndex(std::string const& name) const;
            
            /*!
             * Retrieves the mapping from action names to their indices.
             */
            std::unordered_map<std::string, uint64_t> const& getActionToIndexMap() const;
            
            /**
             * Adds an action to the model.
             *
             * @return the index for this action.
             */
            uint64_t addAction(Action const& action);
            
            /*!
             * Retrieves the action with the given index.
             */
            Action const& getAction(uint64_t index) const;
            
            /*!
             * Retrieves the actions of the model.
             */
            std::vector<Action> const& getActions() const;
    
            /*!
             *  Builds a map with action indices mapped to their names
             */
            std::map<uint64_t, std::string> getActionIndexToNameMap() const;
            
            /*!
             * Retrieves all non-silent action indices of the model.
             */
            boost::container::flat_set<uint64_t> const& getNonsilentActionIndices() const;
            
            /*!
             * Adds the given constant to the model.
             */
            uint64_t addConstant(Constant const& constant);
            
            /*!
             * Retrieves whether the model has a constant with the given name.
             */
            bool hasConstant(std::string const& name) const;
            
            /*!
             * Retrieves the constants of the model.
             */
            std::vector<Constant> const& getConstants() const;

            /*!
             * Retrieves the constants of the model.
             */
            std::vector<Constant>& getConstants();

            /*!
             * Retrieves the constant with the given name (if any).
             */
            Constant const& getConstant(std::string const& name) const;
            
            /*!
             * Adds the given variable to this model.
             */
            Variable const& addVariable(Variable const& variable);

            /*!
             * Adds the given boolean variable to this model.
             */
            BooleanVariable const& addVariable(BooleanVariable const& variable);
            
            /*!
             * Adds the given bounded integer variable to this model.
             */
            BoundedIntegerVariable const& addVariable(BoundedIntegerVariable const& variable);
            
            /*!
             * Adds the given unbounded integer variable to this model.
             */
            UnboundedIntegerVariable const& addVariable(UnboundedIntegerVariable const& variable);

            /*!
             * Adds the given real variable to this model.
             */
            RealVariable const& addVariable(RealVariable const& variable);

            /*!
             * Retrieves the variables of this automaton.
             */
            VariableSet& getGlobalVariables();

            /*!
             * Retrieves the variables of this automaton.
             */
            VariableSet const& getGlobalVariables() const;
            
            /*!
             * Retrieves all expression variables used by this model.
             *
             * @return The set of expression variables used by this model.
             */
            std::set<storm::expressions::Variable> getAllExpressionVariables() const;

            /*!
             * Retrieves whether this model has a global variable with the given name.
             */
            bool hasGlobalVariable(std::string const& name) const;
            
            /*!
             * Retrieves the global variable with the given name if one exists.
             */
            Variable const& getGlobalVariable(std::string const& name) const;
            
            /*!
             * Retrieves whether this model has a non-global transient variable.
             */
            bool hasNonGlobalTransientVariable() const;
            
            /*!
             * Retrieves the manager responsible for the expressions in the JANI model.
             */
            storm::expressions::ExpressionManager& getExpressionManager();

            /*!
             * Retrieves the manager responsible for the expressions in the JANI model.
             */
            storm::expressions::ExpressionManager const& getExpressionManager() const;
            
            /*!
             * Adds the given automaton to the automata of this model.
             */
            uint64_t addAutomaton(Automaton const& automaton);
            
            /*!
             * Retrieves the automata of the model.
             */
            std::vector<Automaton>& getAutomata();

            /*!
             * Retrieves the automata of the model.
             */
            std::vector<Automaton> const& getAutomata() const;

            /*!
             * Retrieves the automaton with the given name.
             */
            Automaton& getAutomaton(std::string const& name);

            /*!
             * Retrieves the automaton with the given name.
             */
            Automaton const& getAutomaton(std::string const& name) const;
            
            /*!
             * Retrieves the index of the given automaton.
             */
            uint64_t getAutomatonIndex(std::string const& name) const;

            /*!
             * Retrieves the number of automata in this model.
             */
            std::size_t getNumberOfAutomata() const;
            
            /*!
             * Sets the system composition expression of the JANI model.
             */
            void setSystemComposition(std::shared_ptr<Composition> const& composition);
            
            /*!
             * Sets the system composition to be the fully-synchronizing parallel composition of all automat
             * @see getStandardSystemComposition
             */
            void setStandardSystemComposition();
            
            /*!
             * Gets the system composition as the standard, fully-synchronizing parallel composition.
             */
            std::shared_ptr<Composition> getStandardSystemComposition() const;
            
            /*!
             * Retrieves the system composition expression.
             */
            Composition const& getSystemComposition() const;
            
            /*!
             * Retrieves the set of action names.
             */
            std::set<std::string> getActionNames(bool includeSilent = true) const;
            
            /*!
             * Defines the undefined constants of the model by the given expressions. The original model is not modified,
             * but instead a new model is created.
             */
            Model defineUndefinedConstants(std::map<storm::expressions::Variable, storm::expressions::Expression> const& constantDefinitions) const;
            
            /*!
             * Retrieves whether the model still has undefined constants.
             */
            bool hasUndefinedConstants() const;
            
            /*!
             * Retrieves all undefined constants of the model.
             */
            std::vector<std::reference_wrapper<Constant const>> getUndefinedConstants() const;
            
            /*!
             * Substitutes all constants in all expressions of the model. The original model is not modified, but
             * instead a new model is created.
             */
            Model substituteConstants() const;
            
            /*!
             * Retrieves a mapping from expression variables associated with defined constants of the model to their
             * (the constants') defining expression.
             */
            std::map<storm::expressions::Variable, storm::expressions::Expression> getConstantsSubstitution() const;
            
            /*!
             * Retrieves whether there is an expression restricting the legal initial values of the global variables.
             */
            bool hasInitialStatesRestriction() const;
            
            /*!
             * Sets the expression restricting the legal initial values of the global variables.
             */
            void setInitialStatesRestriction(storm::expressions::Expression const& initialStatesRestriction);

            /*!
             * Gets the expression restricting the legal initial values of the global variables.
             */
            storm::expressions::Expression const& getInitialStatesRestriction() const;
            
            /*!
             * Retrieves the expression defining the legal initial values of the variables.
             *
             * @param automata The resulting expression will also characterize the legal initial states for these automata.
             */
            storm::expressions::Expression getInitialStatesExpression(std::vector<std::reference_wrapper<storm::jani::Automaton const>> const& automata = {}) const;
            
            /*!
             * Determines whether this model is a deterministic one in the sense that each state only has one choice.
             */
            bool isDeterministicModel() const;
            
            /*!
             * Determines whether this model is a discrete-time model.
             */
            bool isDiscreteTimeModel() const;
            
            /*!
             * Retrieves a list of expressions that characterize the legal values of the variables in this model.
             *
             * @param automata If provided only range expressions from these automata will be created.
             */
            std::vector<storm::expressions::Expression> getAllRangeExpressions(std::vector<std::reference_wrapper<storm::jani::Automaton const>> const& automata = {}) const;
            
            /*!
             * Retrieves whether this model has the standard composition, that is it composes all automata in parallel
             * and synchronizes over all common actions.
             */
            bool hasStandardComposition() const;
            
            /*!
             * Checks whether the composition has no nesting.
             */
            bool hasStandardCompliantComposition() const;
            
            /*!
             * After adding all components to the model, this method has to be called. It recursively calls
             * <code>finalize</code> on all contained elements. All subsequent changes to the model require another call
             * to this method.
             */
            void finalize();
            
            /*!
             *  Checks if the model is valid JANI, which should be verified before any further operations are applied to a model.
             */
            void checkValid() const;
            
            /*!
             * Creates the expression that characterizes all states in which the provided transient boolean variable is
             * true. The provided location variables are used to encode the location of the automata.
             */
            storm::expressions::Expression getLabelExpression(BooleanVariable const& transientVariable, std::map<std::string, storm::expressions::Variable> const& automatonToLocationVariableMap) const;
            
            /*!
             * Checks that undefined constants (parameters) of the model preserve the graph of the underlying model.
             * That is, undefined constants may only appear in the probability expressions of edge destinations as well
             * as on the right-hand sides of transient assignments.
             */
            bool undefinedConstantsAreGraphPreserving() const;
            
            /*!
             * Lifts the common edge destination assignments to edge assignments.
             */
            void liftTransientEdgeDestinationAssignments();
            
            /*!
             * Retrieves whether there is any transient edge destination assignment in the model.
             */
            bool hasTransientEdgeDestinationAssignments() const;
            
            /*!
             * Retrieves whether the model uses an assignment level other than zero.
             */
            bool usesAssignmentLevels() const;
            
            void makeStandardJaniCompliant();
            
            /// The name of the silent action.
            static const std::string SILENT_ACTION_NAME;
            
            /// The index of the silent action.
            static const uint64_t SILENT_ACTION_INDEX;
            
        private:
            /*!
             * Creates a new model from the given automaton (which must be contained in the current model).
             */
            Model createModelFromAutomaton(Automaton const& automaton) const;
                        
            /// The model name.
            std::string name;
            
            /// The type of the model.
            ModelType modelType;
            
            /// The JANI-version used to specify the model.
            uint64_t version;
            
            /// The manager responsible for the expressions in this model.
            std::shared_ptr<storm::expressions::ExpressionManager> expressionManager;
            
            /// The list of actions.
            std::vector<Action> actions;
            
            /// A mapping from names to action indices.
            std::unordered_map<std::string, uint64_t> actionToIndex;
            
            /// The set of non-silent action indices.
            boost::container::flat_set<uint64_t> nonsilentActionIndices;
            
            /// The constants defined by the model.
            std::vector<Constant> constants;
            
            /// A mapping from names to constants.
            std::unordered_map<std::string, uint64_t> constantToIndex;
            
            /// The global variables of the model.
            VariableSet globalVariables;
            
            /// The list of automata.
            std::vector<Automaton> automata;
            
            /// A mapping from names to automata indices.
            std::unordered_map<std::string, size_t> automatonToIndex;
            
            /// An expression describing how the system is composed of the automata.
            std::shared_ptr<Composition> composition;
            
            // The expression restricting the legal initial values of the global variables.
            storm::expressions::Expression initialStatesRestriction;
        };
    }
}

