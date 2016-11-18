#pragma once

#include <vector>
#include <set>

#include "storm/adapters/DereferenceIteratorAdapter.h"

#include "storm/storage/jani/BooleanVariable.h"
#include "storm/storage/jani/UnboundedIntegerVariable.h"
#include "storm/storage/jani/BoundedIntegerVariable.h"
#include "storm/storage/jani/RealVariable.h"

namespace storm {
    namespace jani {
                
        namespace detail {
            template <typename VariableType>
            using Variables = storm::adapters::DereferenceIteratorAdapter<std::vector<std::shared_ptr<VariableType>>>;

            template <typename VariableType>
            using ConstVariables = storm::adapters::DereferenceIteratorAdapter<std::vector<std::shared_ptr<VariableType>> const>;
        }
        
        class VariableSet {
        public:
            /*!
             * Creates an empty variable set.
             */
            VariableSet();
            
            /*!
             * Retrieves the boolean variables in this set.
             */
            detail::Variables<BooleanVariable> getBooleanVariables();

            /*!
             * Retrieves the boolean variables in this set.
             */
            detail::ConstVariables<BooleanVariable> getBooleanVariables() const;

            /*!
             * Retrieves the bounded integer variables in this set.
             */
            detail::Variables<BoundedIntegerVariable> getBoundedIntegerVariables();

            /*!
             * Retrieves the bounded integer variables in this set.
             */
            detail::ConstVariables<BoundedIntegerVariable> getBoundedIntegerVariables() const;

            /*!
             * Retrieves the unbounded integer variables in this set.
             */
            detail::Variables<UnboundedIntegerVariable> getUnboundedIntegerVariables();

            /*!
             * Retrieves the unbounded integer variables in this set.
             */
            detail::ConstVariables<UnboundedIntegerVariable> getUnboundedIntegerVariables() const;
            
            /*!
             * Retrieves the real variables in this set.
             */
            detail::Variables<RealVariable> getRealVariables();
            
            /*!
             * Retrieves the real variables in this set.
             */
            detail::ConstVariables<RealVariable> getRealVariables() const;
            
            /*!
             * Adds the given boolean variable to this set.
             */
            BooleanVariable const& addVariable(BooleanVariable const& variable);

            /*!
             * Adds the given bounded integer variable to this set.
             */
            BoundedIntegerVariable const& addVariable(BoundedIntegerVariable const& variable);

            /*!
             * Adds the given unbounded integer variable to this set.
             */
            UnboundedIntegerVariable const& addVariable(UnboundedIntegerVariable const& variable);
            
            /*!
             * Adds the given real variable to this set.
             */
            RealVariable const& addVariable(RealVariable const& variable);

            /*!
             * Retrieves whether this variable set contains a variable with the given name.
             */
            bool hasVariable(std::string const& name) const;
            
            /*!
             * Retrieves the variable with the given name.
             */
            Variable const& getVariable(std::string const& name) const;

            /*!
             * Retrieves whether this variable set contains a variable with the expression variable.
             */
            bool hasVariable(storm::expressions::Variable const& variable) const;
            
            /*!
             * Retrieves the variable object associated with the given expression variable (if any).
             */
            Variable const& getVariable(storm::expressions::Variable const& variable) const;

            /*!
             * Retrieves whether this variable set contains a transient variable.
             */
            bool hasTransientVariable() const;
            
            /*!
             * Retrieves an iterator to the variables in this set.
             */
            typename detail::Variables<Variable>::iterator begin();

            /*!
             * Retrieves an iterator to the variables in this set.
             */
            typename detail::ConstVariables<Variable>::iterator begin() const;

            /*!
             * Retrieves the end iterator to the variables in this set.
             */
            typename detail::Variables<Variable>::iterator end();

            /*!
             * Retrieves the end iterator to the variables in this set.
             */
            typename detail::ConstVariables<Variable>::iterator end() const;

            /*!
             * Retrieves whether the set of variables contains a boolean variable.
             */
            bool containsBooleanVariable() const;

            /*!
             * Retrieves whether the set of variables contains a bounded integer variable.
             */
            bool containsBoundedIntegerVariable() const;

            /*!
             * Retrieves whether the set of variables contains an unbounded integer variable.
             */
            bool containsUnboundedIntegerVariables() const;

            /*!
             * Retrieves whether the set of variables contains a real variable.
             */
            bool containsRealVariables() const;

            /*!
             * Retrieves whether the set of variables contains a non-transient real variable.
             */
            bool containsNonTransientRealVariables() const;

            /*!
             * Retrieves whether the set of variables contains a non-transient unbounded integer variable.
             */
            bool containsNonTransientUnboundedIntegerVariables() const;

            /*!
             * Retrieves whether this variable set is empty.
             */
            bool empty() const;
            
            /*!
             * Retrieves the number of transient variables in this variable set.
             */
            uint_fast64_t getNumberOfTransientVariables() const;
            
            /*!
             * Retrieves the number of real transient variables in this variable set.
             */
            uint_fast64_t getNumberOfRealTransientVariables() const;

            /*!
             * Retrieves the number of unbounded integer transient variables in this variable set.
             */
            uint_fast64_t getNumberOfUnboundedIntegerTransientVariables() const;

            /*!
             * Retrieves the transient variables in this variable set.
             */
            typename detail::ConstVariables<Variable> getTransientVariables() const;
            
            /*!
             * Checks whether any of the provided variables appears in bound expressions or initial values of the
             * variables contained in this variable set.
             */
            bool containsVariablesInBoundExpressionsOrInitialValues(std::set<storm::expressions::Variable> const& variables) const;
            
        private:
            /// The vector of all variables.
            std::vector<std::shared_ptr<Variable>> variables;
            
            /// The boolean variables in this set.
            std::vector<std::shared_ptr<BooleanVariable>> booleanVariables;

            /// The bounded integer variables in this set.
            std::vector<std::shared_ptr<BoundedIntegerVariable>> boundedIntegerVariables;

            /// The unbounded integer variables in this set.
            std::vector<std::shared_ptr<UnboundedIntegerVariable>> unboundedIntegerVariables;
            
            /// The real variables in this set.
            std::vector<std::shared_ptr<RealVariable>> realVariables;
            
            /// The transient variables in this set.
            std::vector<std::shared_ptr<Variable>> transientVariables;
            
            /// A set of all variable names currently in use.
            std::map<std::string, storm::expressions::Variable> nameToVariable;
            
            /// A mapping from expression variables to their variable objects.
            std::map<storm::expressions::Variable, std::shared_ptr<Variable>> variableToVariable;
        };
        
    }
}
