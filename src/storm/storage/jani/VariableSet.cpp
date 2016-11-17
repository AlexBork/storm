#include "src/storm/storage/jani/VariableSet.h"

#include "src/storm/utility/macros.h"
#include "src/storm/exceptions/WrongFormatException.h"
#include "src/storm/exceptions/InvalidArgumentException.h"

namespace storm {
    namespace jani {
                
        VariableSet::VariableSet() {
            // Intentionally left empty.
        }
        
        detail::Variables<BooleanVariable> VariableSet::getBooleanVariables() {
            return detail::Variables<BooleanVariable>(booleanVariables.begin(), booleanVariables.end());
        }
        
        detail::ConstVariables<BooleanVariable> VariableSet::getBooleanVariables() const {
            return detail::ConstVariables<BooleanVariable>(booleanVariables.begin(), booleanVariables.end());
        }

        detail::Variables<BoundedIntegerVariable> VariableSet::getBoundedIntegerVariables() {
            return detail::Variables<BoundedIntegerVariable>(boundedIntegerVariables.begin(), boundedIntegerVariables.end());
        }

        detail::ConstVariables<BoundedIntegerVariable> VariableSet::getBoundedIntegerVariables() const {
            return detail::ConstVariables<BoundedIntegerVariable>(boundedIntegerVariables.begin(), boundedIntegerVariables.end());
        }

        detail::Variables<UnboundedIntegerVariable> VariableSet::getUnboundedIntegerVariables() {
            return detail::Variables<UnboundedIntegerVariable>(unboundedIntegerVariables.begin(), unboundedIntegerVariables.end());
        }

        detail::ConstVariables<UnboundedIntegerVariable> VariableSet::getUnboundedIntegerVariables() const {
            return detail::ConstVariables<UnboundedIntegerVariable>(unboundedIntegerVariables.begin(), unboundedIntegerVariables.end());
        }

        detail::Variables<RealVariable> VariableSet::getRealVariables() {
            return detail::Variables<RealVariable>(realVariables.begin(), realVariables.end());
        }
        
        detail::ConstVariables<RealVariable> VariableSet::getRealVariables() const {
            return detail::ConstVariables<RealVariable>(realVariables.begin(), realVariables.end());
        }
        
        BooleanVariable const& VariableSet::addVariable(BooleanVariable const& variable) {
            STORM_LOG_THROW(!this->hasVariable(variable.getName()), storm::exceptions::WrongFormatException, "Cannot add variable with name '" << variable.getName() << "', because a variable with that name already exists.");
            std::shared_ptr<BooleanVariable> newVariable = std::make_shared<BooleanVariable>(variable);
            variables.push_back(newVariable);
            booleanVariables.push_back(newVariable);
            if (variable.isTransient()) {
                transientVariables.push_back(newVariable);
            }
            nameToVariable.emplace(variable.getName(), variable.getExpressionVariable());
            variableToVariable.emplace(variable.getExpressionVariable(), newVariable);
            return *newVariable;
        }
        
        BoundedIntegerVariable const& VariableSet::addVariable(BoundedIntegerVariable const& variable) {
            STORM_LOG_THROW(!this->hasVariable(variable.getName()), storm::exceptions::WrongFormatException, "Cannot add variable with name '" << variable.getName() << "', because a variable with that name already exists.");
            std::shared_ptr<BoundedIntegerVariable> newVariable = std::make_shared<BoundedIntegerVariable>(variable);
            variables.push_back(newVariable);
            boundedIntegerVariables.push_back(newVariable);
            if (variable.isTransient()) {
                transientVariables.push_back(newVariable);
            }
            nameToVariable.emplace(variable.getName(), variable.getExpressionVariable());
            variableToVariable.emplace(variable.getExpressionVariable(), newVariable);
            return *newVariable;
        }
        
        UnboundedIntegerVariable const& VariableSet::addVariable(UnboundedIntegerVariable const& variable) {
            STORM_LOG_THROW(!this->hasVariable(variable.getName()), storm::exceptions::WrongFormatException, "Cannot add variable with name '" << variable.getName() << "', because a variable with that name already exists.");
            std::shared_ptr<UnboundedIntegerVariable> newVariable = std::make_shared<UnboundedIntegerVariable>(variable);
            variables.push_back(newVariable);
            unboundedIntegerVariables.push_back(newVariable);
            if (variable.isTransient()) {
                transientVariables.push_back(newVariable);
            }
            nameToVariable.emplace(variable.getName(), variable.getExpressionVariable());
            variableToVariable.emplace(variable.getExpressionVariable(), newVariable);
            return *newVariable;
        }
        
        RealVariable const& VariableSet::addVariable(RealVariable const& variable) {
            STORM_LOG_THROW(!this->hasVariable(variable.getName()), storm::exceptions::WrongFormatException, "Cannot add variable with name '" << variable.getName() << "', because a variable with that name already exists.");
            std::shared_ptr<RealVariable> newVariable = std::make_shared<RealVariable>(variable);
            variables.push_back(newVariable);
            realVariables.push_back(newVariable);
            if (variable.isTransient()) {
                transientVariables.push_back(newVariable);
            }
            nameToVariable.emplace(variable.getName(), variable.getExpressionVariable());
            variableToVariable.emplace(variable.getExpressionVariable(), newVariable);
            return *newVariable;
        }
        
        bool VariableSet::hasVariable(std::string const& name) const {
            return nameToVariable.find(name) != nameToVariable.end();
        }
        
        Variable const& VariableSet::getVariable(std::string const& name) const {
            auto it = nameToVariable.find(name);
            STORM_LOG_THROW(it != nameToVariable.end(), storm::exceptions::InvalidArgumentException, "Unable to retrieve unknown variable '" << name << "'.");
            return getVariable(it->second);
        }

        typename detail::Variables<Variable>::iterator VariableSet::begin() {
            return detail::Variables<Variable>::make_iterator(variables.begin());
        }

        typename detail::ConstVariables<Variable>::iterator VariableSet::begin() const {
            return detail::ConstVariables<Variable>::make_iterator(variables.begin());
        }
        
        typename detail::Variables<Variable>::iterator VariableSet::end() {
            return detail::Variables<Variable>::make_iterator(variables.end());
        }

        detail::ConstVariables<Variable>::iterator VariableSet::end() const {
            return detail::ConstVariables<Variable>::make_iterator(variables.end());
        }

        Variable const& VariableSet::getVariable(storm::expressions::Variable const& variable) const {
            auto it = variableToVariable.find(variable);
            STORM_LOG_THROW(it != variableToVariable.end(), storm::exceptions::InvalidArgumentException, "Unable to retrieve unknown variable '" << variable.getName() << "'.");
            return *it->second;
        }
        
        bool VariableSet::hasVariable(storm::expressions::Variable const& variable) const {
            return variableToVariable.find(variable) != variableToVariable.end();
        }
        
        bool VariableSet::hasTransientVariable() const {
            for (auto const& variable : variables) {
                if (variable->isTransient()) {
                    return true;
                }
            }
            return false;
        }
        
        bool VariableSet::containsBooleanVariable() const {
            return !booleanVariables.empty();
        }
        
        bool VariableSet::containsBoundedIntegerVariable() const {
            return !boundedIntegerVariables.empty();
        }
        
        bool VariableSet::containsUnboundedIntegerVariables() const {
            return !unboundedIntegerVariables.empty();
        }
        
        bool VariableSet::containsRealVariables() const {
            return !realVariables.empty();
        }
        
        bool VariableSet::containsNonTransientRealVariables() const {
            for (auto const& variable : realVariables) {
                if (!variable->isTransient()) {
                    return true;
                }
            }
            return false;
        }
        
        bool VariableSet::containsNonTransientUnboundedIntegerVariables() const {
            for (auto const& variable : unboundedIntegerVariables) {
                if (!variable->isTransient()) {
                    return true;
                }
            }
            return false;
        }
        
        bool VariableSet::empty() const {
            return !(containsBooleanVariable() || containsBoundedIntegerVariable() || containsUnboundedIntegerVariables());
        }
        
        uint_fast64_t VariableSet::getNumberOfTransientVariables() const {
            uint_fast64_t result = 0;
            for (auto const& variable : variables) {
                if (variable->isTransient()) {
                    ++result;
                }
            }
            return result;
        }
        
        uint_fast64_t VariableSet::getNumberOfRealTransientVariables() const {
            uint_fast64_t result = 0;
            for (auto const& variable : variables) {
                if (variable->isTransient() && variable->isRealVariable()) {
                    ++result;
                }
            }
            return result;
        }
        
        uint_fast64_t VariableSet::getNumberOfUnboundedIntegerTransientVariables() const {
            uint_fast64_t result = 0;
            for (auto const& variable : variables) {
                if (variable->isTransient() && variable->isUnboundedIntegerVariable()) {
                    ++result;
                }
            }
            return result;
        }
        
        typename detail::ConstVariables<Variable> VariableSet::getTransientVariables() const {
            return detail::ConstVariables<Variable>(transientVariables.begin(), transientVariables.end());
        }
        
        bool VariableSet::containsVariablesInBoundExpressionsOrInitialValues(std::set<storm::expressions::Variable> const& variables) const {
            for (auto const& booleanVariable : this->getBooleanVariables()) {
                if (booleanVariable.hasInitExpression()) {
                    if (booleanVariable.getInitExpression().containsVariable(variables)) {
                        return true;
                    }
                }
            }
            for (auto const& integerVariable : this->getBoundedIntegerVariables()) {
                if (integerVariable.hasInitExpression()) {
                    if (integerVariable.getInitExpression().containsVariable(variables)) {
                        return true;
                    }
                }
                if (integerVariable.getLowerBound().containsVariable(variables)) {
                    return true;
                }
                if (integerVariable.getUpperBound().containsVariable(variables)) {
                    return true;
                }
            }
            return false;
        }
        
    }
}
