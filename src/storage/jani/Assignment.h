#pragma once

#include <functional>

#include "src/storage/jani/Variable.h"
#include "src/storage/expressions/Expression.h"

namespace storm {
    namespace jani {
        
        class Assignment {
        public:
            /*!
             * Creates an assignment of the given expression to the given variable.
             */
            Assignment(storm::jani::Variable const& variable, storm::expressions::Expression const& expression, uint64_t index = 0);
            
            bool operator==(Assignment const& other) const;
            
            /*!
             * Retrieves the expression variable that is written in this assignment.
             */
            storm::jani::Variable const& getVariable() const;
                
            /*!
             * Retrieves the expression variable that is written in this assignment.
             */
            storm::expressions::Variable const& getExpressionVariable() const;
            
            /*!
             * Retrieves the expression whose value is assigned to the target variable.
             */
            storm::expressions::Expression const& getAssignedExpression() const;
            
            /*!
             * Sets a new expression that is assigned to the target variable.
             */
            void setAssignedExpression(storm::expressions::Expression const& expression);

            /*!
             * Substitutes all variables in all expressions according to the given substitution.
             */
            void substitute(std::map<storm::expressions::Variable, storm::expressions::Expression> const& substitution);
            
            /**
             * Retrieves whether the assignment assigns to a transient variable.
             */
            bool isTransient() const;
            
            /*!
             * Retrieves the level of the assignment.
             */
            int64_t getLevel() const;
            
            friend std::ostream& operator<<(std::ostream& stream, Assignment const& assignment);
            
        private:
            // The variable being assigned.
            std::reference_wrapper<storm::jani::Variable const> variable;
            
            // The expression that is being assigned to the variable.
            storm::expressions::Expression expression;
            
            // The level of the assignment.
            uint64_t level;
        };
        
        /*!
         * This functor enables ordering the assignments first by the assignment level and then by variable.
         * Note that this is a partial order.
         */
        struct AssignmentPartialOrderByLevelAndVariable {
            bool operator()(Assignment const& left, Assignment const& right) const;
            bool operator()(Assignment const& left, std::shared_ptr<Assignment> const& right) const;
            bool operator()(std::shared_ptr<Assignment> const& left, std::shared_ptr<Assignment> const& right) const;
            bool operator()(std::shared_ptr<Assignment> const& left, Assignment const& right) const;
        };
    
    }
}
